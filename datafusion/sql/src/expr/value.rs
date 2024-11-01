// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::planner::{ContextProvider, PlannerContext, SqlToRel};
use arrow::compute::kernels::cast_utils::{
    parse_interval_month_day_nano_config, IntervalParseConfig, IntervalUnit,
};
use arrow::datatypes::DECIMAL128_MAX_PRECISION;
use arrow_schema::{DataType, DECIMAL128_MAX_SCALE};
use bigdecimal::ToPrimitive;
use datafusion_common::{
    internal_err, not_impl_err, plan_err, DFSchema, DataFusionError, Result, ScalarValue,
};
use datafusion_expr::expr::{BinaryExpr, Placeholder};
use datafusion_expr::planner::PlannerResult;
use datafusion_expr::{lit, Expr, Operator};
use log::debug;
use sqlparser::ast::{BinaryOperator, Expr as SQLExpr, Interval, UnaryOperator, Value};
use sqlparser::parser::ParserError::ParserError;
use std::borrow::Cow;
use std::ops::Neg;
use std::str::FromStr;

impl<'a, S: ContextProvider> SqlToRel<'a, S> {
    pub(crate) fn parse_value(
        &self,
        value: Value,
        param_data_types: &[DataType],
    ) -> Result<Expr> {
        match value {
            Value::Number(n, _) => self.parse_sql_number(&n, false),
            Value::SingleQuotedString(s) | Value::DoubleQuotedString(s) => Ok(lit(s)),
            Value::Null => Ok(Expr::Literal(ScalarValue::Null)),
            Value::Boolean(n) => Ok(lit(n)),
            Value::Placeholder(param) => {
                Self::create_placeholder_expr(param, param_data_types)
            }
            Value::HexStringLiteral(s) => {
                if let Some(v) = try_decode_hex_literal(&s) {
                    Ok(lit(v))
                } else {
                    plan_err!("Invalid HexStringLiteral '{s}'")
                }
            }
            Value::DollarQuotedString(s) => Ok(lit(s.value)),
            Value::EscapedStringLiteral(s) => Ok(lit(s)),
            _ => plan_err!("Unsupported Value '{value:?}'"),
        }
    }

    /// Parse number in sql string, convert to Expr::Literal
    pub(super) fn parse_sql_number(
        &self,
        unsigned_number: &str,
        negative: bool,
    ) -> Result<Expr> {
        let signed_number: Cow<str> = if negative {
            Cow::Owned(format!("-{unsigned_number}"))
        } else {
            Cow::Borrowed(unsigned_number)
        };

        // Try to parse as i64 first, then u64 if negative is false, then decimal or f64

        if let Ok(n) = signed_number.parse::<i64>() {
            return Ok(lit(n));
        }

        if !negative {
            if let Ok(n) = unsigned_number.parse::<u64>() {
                return Ok(lit(n));
            }
        }

        if self.options.parse_float_as_decimal {
            parse_decimal_128(unsigned_number, negative)
        } else {
            signed_number.parse::<f64>().map(lit).map_err(|_| {
                DataFusionError::from(ParserError(format!(
                    "Cannot parse {signed_number} as f64"
                )))
            })
        }
    }

    /// Create a placeholder expression
    /// This is the same as Postgres's prepare statement syntax in which a placeholder starts with `$` sign and then
    /// number 1, 2, ... etc. For example, `$1` is the first placeholder; $2 is the second one and so on.
    fn create_placeholder_expr(
        param: String,
        param_data_types: &[DataType],
    ) -> Result<Expr> {
        // Parse the placeholder as a number because it is the only support from sqlparser and postgres
        let index = param[1..].parse::<usize>();
        let idx = match index {
            Ok(0) => {
                return plan_err!(
                    "Invalid placeholder, zero is not a valid index: {param}"
                );
            }
            Ok(index) => index - 1,
            Err(_) => {
                return if param_data_types.is_empty() {
                    Ok(Expr::Placeholder(Placeholder::new(param, None)))
                } else {
                    // when PREPARE Statement, param_data_types length is always 0
                    plan_err!("Invalid placeholder, not a number: {param}")
                };
            }
        };
        // Check if the placeholder is in the parameter list
        let param_type = param_data_types.get(idx);
        // Data type of the parameter
        debug!(
            "type of param {} param_data_types[idx]: {:?}",
            param, param_type
        );

        Ok(Expr::Placeholder(Placeholder::new(
            param,
            param_type.cloned(),
        )))
    }

    pub(super) fn sql_array_literal(
        &self,
        elements: Vec<SQLExpr>,
        schema: &DFSchema,
    ) -> Result<Expr> {
        let values = elements
            .into_iter()
            .map(|element| {
                self.sql_expr_to_logical_expr(element, schema, &mut PlannerContext::new())
            })
            .collect::<Result<Vec<_>>>()?;

        self.try_plan_array_literal(values, schema)
    }

    fn try_plan_array_literal(
        &self,
        values: Vec<Expr>,
        schema: &DFSchema,
    ) -> Result<Expr> {
        let mut exprs = values;
        for planner in self.context_provider.get_expr_planners() {
            match planner.plan_array_literal(exprs, schema)? {
                PlannerResult::Planned(expr) => {
                    return Ok(expr);
                }
                PlannerResult::Original(values) => exprs = values,
            }
        }

        internal_err!("Expected a simplified result, but none was found")
    }

    /// Convert a SQL interval expression to a DataFusion logical plan
    /// expression
    #[allow(clippy::only_used_in_recursion)]
    pub(super) fn sql_interval_to_expr(
        &self,
        negative: bool,
        interval: Interval,
    ) -> Result<Expr> {
        if interval.leading_precision.is_some() {
            return not_impl_err!(
                "Unsupported Interval Expression with leading_precision {:?}",
                interval.leading_precision
            );
        }

        if interval.last_field.is_some() {
            return not_impl_err!(
                "Unsupported Interval Expression with last_field {:?}",
                interval.last_field
            );
        }

        if interval.fractional_seconds_precision.is_some() {
            return not_impl_err!(
                "Unsupported Interval Expression with fractional_seconds_precision {:?}",
                interval.fractional_seconds_precision
            );
        }

        if let SQLExpr::BinaryOp { left, op, right } = *interval.value {
            let df_op = match op {
                BinaryOperator::Plus => Operator::Plus,
                BinaryOperator::Minus => Operator::Minus,
                _ => {
                    return not_impl_err!("Unsupported interval operator: {op:?}");
                }
            };
            let left_expr = self.sql_interval_to_expr(
                negative,
                Interval {
                    value: left,
                    leading_field: interval.leading_field.clone(),
                    leading_precision: None,
                    last_field: None,
                    fractional_seconds_precision: None,
                },
            )?;
            let right_expr = self.sql_interval_to_expr(
                false,
                Interval {
                    value: right,
                    leading_field: interval.leading_field,
                    leading_precision: None,
                    last_field: None,
                    fractional_seconds_precision: None,
                },
            )?;
            return Ok(Expr::BinaryExpr(BinaryExpr::new(
                Box::new(left_expr),
                df_op,
                Box::new(right_expr),
            )));
        }

        let value = interval_literal(*interval.value, negative)?;

        // leading_field really means the unit if specified
        // For example, "month" in  `INTERVAL '5' month`
        let value = match interval.leading_field.as_ref() {
            Some(leading_field) => format!("{value} {leading_field}"),
            None => value,
        };

        let config = IntervalParseConfig::new(IntervalUnit::Second);
        let val = parse_interval_month_day_nano_config(&value, config)?;
        Ok(lit(ScalarValue::IntervalMonthDayNano(Some(val))))
    }
}

fn interval_literal(interval_value: SQLExpr, negative: bool) -> Result<String> {
    let s = match interval_value {
        SQLExpr::Value(Value::SingleQuotedString(s) | Value::DoubleQuotedString(s)) => s,
        SQLExpr::Value(Value::Number(ref v, long)) => {
            if long {
                return not_impl_err!(
                    "Unsupported interval argument. Long number not supported: {interval_value:?}"
                );
            } else {
                v.to_string()
            }
        }
        SQLExpr::UnaryOp { op, expr } => {
            let negative = match op {
                UnaryOperator::Minus => !negative,
                UnaryOperator::Plus => negative,
                _ => {
                    return not_impl_err!(
                        "Unsupported SQL unary operator in interval {op:?}"
                    );
                }
            };
            interval_literal(*expr, negative)?
        }
        _ => {
            return not_impl_err!("Unsupported interval argument. Expected string literal or number, got: {interval_value:?}");
        }
    };
    if negative {
        Ok(format!("-{s}"))
    } else {
        Ok(s)
    }
}

/// Try to decode bytes from hex literal string.
///
/// None will be returned if the input literal is hex-invalid.
fn try_decode_hex_literal(s: &str) -> Option<Vec<u8>> {
    let hex_bytes = s.as_bytes();

    let mut decoded_bytes = Vec::with_capacity((hex_bytes.len() + 1) / 2);

    let start_idx = hex_bytes.len() % 2;
    if start_idx > 0 {
        // The first byte is formed of only one char.
        decoded_bytes.push(try_decode_hex_char(hex_bytes[0])?);
    }

    for i in (start_idx..hex_bytes.len()).step_by(2) {
        let high = try_decode_hex_char(hex_bytes[i])?;
        let low = try_decode_hex_char(hex_bytes[i + 1])?;
        decoded_bytes.push(high << 4 | low);
    }

    Some(decoded_bytes)
}

/// Try to decode a byte from a hex char.
///
/// None will be returned if the input char is hex-invalid.
const fn try_decode_hex_char(c: u8) -> Option<u8> {
    match c {
        b'A'..=b'F' => Some(c - b'A' + 10),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'0'..=b'9' => Some(c - b'0'),
        _ => None,
    }
}

/// Parse Decimal128 from a string
///
/// TODO: support parsing from scientific notation
fn parse_decimal_128(unsigned_number: &str, negative: bool) -> Result<Expr> {
    let big_decimal = bigdecimal::BigDecimal::from_str(unsigned_number).map_err(|e| {
        DataFusionError::from(ParserError(format!(
            "Cannot parse {unsigned_number} as i128 when building decimal: {e}"
        )))
    })?;

    let (bigint, scale) = big_decimal.as_bigint_and_exponent();
    let digits = big_decimal.digits();

    if digits > DECIMAL128_MAX_PRECISION as u64 {
        return Err(DataFusionError::from(ParserError(format!(
            "Cannot parse {bigint} as i128 when building decimal: precision overflow"
        ))));
    }

    if scale.unsigned_abs() > DECIMAL128_MAX_SCALE as u64 {
        return Err(DataFusionError::from(ParserError(format!(
            "Cannot parse {unsigned_number} as i128 when building decimal: scale overflow"
        ))));
    }

    // This is a workaround for encoding values where the scale is
    // greater than the precision.
    //
    // Example #1: `0000.00`
    // The leading zeroes are not significant so this is the same as
    // the value `0.00`. The precision is `1` and the scale is `2`.
    //
    // Runtime error:
    // `Arrow error: Invalid argument error: scale 2 is greater than precision 1`
    //
    // Example #2: `0.01`
    // This is the same as `1x10^(-2)` or `1E-2`. The precision is `1`
    // and the scale is `2`.
    //
    // Runtime error:
    // `Arrow error: Invalid argument error: scale 2 is greater than precision 1`
    //
    // Example #3: `0.011`
    // This is the same as `11x10^(-3)` or `11E-3`. The precision is
    // `2` and the scale is `3`.
    //
    // Runtime error:
    // `Arrow error: Invalid argument error: scale 3 is greater than precision 2`
    //
    let precision = if scale > digits as i64 {
        scale as u64
    } else {
        digits
    };

    // It is safe to `.unwrap()` because `i128::MAX` and `i128::MIN` are
    // 39 digits and we have already verified that precision does not
    // exceed `DECIMAL128_MAX_SCALE` which is defined as 38.
    let number = bigint
        .to_i128()
        .map(|value| if negative { value.neg() } else { value })
        .unwrap();
    // .ok_or(DataFusionError::from(ParserError(format!(
    //     "Cannot parse {unsigned_number} as i128 when building decimal"
    // ))))?;

    Ok(Expr::Literal(ScalarValue::Decimal128(
        Some(number),
        precision as u8,
        scale as i8,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_decode_hex_literal() {
        let cases = [
            ("", Some(vec![])),
            ("FF00", Some(vec![255, 0])),
            ("a00a", Some(vec![160, 10])),
            ("FF0", Some(vec![15, 240])),
            ("f", Some(vec![15])),
            ("FF0X", None),
            ("X0", None),
            ("XX", None),
            ("x", None),
        ];

        for (input, expect) in cases {
            let output = try_decode_hex_literal(input);
            assert_eq!(output, expect);
        }
    }
    #[test]
    fn test_decimal_zero_parsing() {
        let zeroes_basic = vec![
            ("0", 1, 0),
            ("00", 1, 0),
            ("000", 1, 0),
            ("0.", 1, 0),
            ("00.", 1, 0),
            ("000.", 1, 0),
            ("0.0", 1, 1),
            ("00.0", 1, 1),
            ("000.0", 1, 1),
            ("0.00", 1, 2),
            ("00.00", 1, 2),
            ("000.00", 1, 2),
            ("0.000", 1, 3),
            ("00.000", 1, 3),
            ("000.000", 1, 3),
            (".0", 1, 1),
            (".00", 1, 2),
            (".000", 1, 3),
        ];

        let zeroes_exp_zero = vec![
            ("0e0", 1, 0),
            ("00e0", 1, 0),
            ("000e0", 1, 0),
            ("0.e0", 1, 0),
            ("00.e0", 1, 0),
            ("000.e0", 1, 0),
            ("0.0e0", 1, 1),
            ("00.0e0", 1, 1),
            ("000.0e0", 1, 1),
            ("0.00e0", 1, 2),
            ("00.00e0", 1, 2),
            ("000.00e0", 1, 2),
            ("0.000e0", 1, 3),
            ("00.000e0", 1, 3),
            ("000.000e0", 1, 3),
            (".0e0", 1, 1),
            (".00e0", 1, 2),
            (".000e0", 1, 3),
        ];

        let zeroes_exp_one = vec![
            ("0e1", 1, -1),
            ("00e1", 1, -1),
            ("000e1", 1, -1),
            ("0.e1", 1, -1),
            ("00.e1", 1, -1),
            ("000.e1", 1, -1),
            ("0.0e1", 1, 0),
            ("00.0e1", 1, 0),
            ("000.0e1", 1, 0),
            ("0.00e1", 1, 1),
            ("00.00e1", 1, 1),
            ("000.00e1", 1, 1),
            ("0.000e1", 1, 2),
            ("00.000e1", 1, 2),
            ("000.000e1", 1, 2),
            (".0e1", 1, 0),
            (".00e1", 1, 1),
            (".000e1", 1, 2),
        ];

        let zeroes_exp_two = vec![
            ("0e2", 1, -2),
            ("00e2", 1, -2),
            ("000e2", 1, -2),
            ("0.e2", 1, -2),
            ("00.e2", 1, -2),
            ("000.e2", 1, -2),
            ("0.0e2", 1, -1),
            ("00.0e2", 1, -1),
            ("000.0e2", 1, -1),
            ("0.00e2", 1, 0),
            ("00.00e2", 1, 0),
            ("000.00e2", 1, 0),
            ("0.000e2", 1, 1),
            ("00.000e2", 1, 1),
            ("000.000e2", 1, 1),
            (".0e2", 1, -1),
            (".00e2", 1, 0),
            (".000e2", 1, 1),
        ];

        let zeroes_exp_three = vec![
            ("0e3", 1, -3),
            ("00e3", 1, -3),
            ("000e3", 1, -3),
            ("0.e3", 1, -3),
            ("00.e3", 1, -3),
            ("000.e3", 1, -3),
            ("0.0e3", 1, -2),
            ("00.0e3", 1, -2),
            ("000.0e3", 1, -2),
            ("0.00e3", 1, -1),
            ("00.00e3", 1, -1),
            ("000.00e3", 1, -1),
            ("0.000e3", 1, 0),
            ("00.000e3", 1, 0),
            ("000.000e3", 1, 0),
            (".0e3", 1, -2),
            (".00e3", 1, -1),
            (".000e3", 1, 0),
        ];

        let other_cases = vec![
            ("0.1", 1, 1),                      // "Decimal128(Some(1),1,1)"),
            ("0.01", 1, 2),                     // "Decimal128(Some(1),2,2)"),
            ("1.0", 2, 1),                      // "Decimal128(Some(10),2,1)"),
            ("10.01", 4, 2),                    // "Decimal128(Some(1001),4,2)"),
            ("10.00", 4, 2),                    // "Decimal128(Some(1001),4,2)"),
            ("10000000000000000000.00", 22, 2), // "Decimal128(Some(1000000000000000000000),22,2)",
            ("18446744073709551616", 20, 0), // "Decimal128(Some(18446744073709551616),20,0)",
        ];

        let test_cases = [
            zeroes_basic,
            zeroes_exp_zero,
            zeroes_exp_one,
            zeroes_exp_two,
            zeroes_exp_three,
            other_cases,
        ]
        .concat();

        // Table header
        println!(" ## |{:8}in |{:18}BD | bgnt | prec | scle | norm |", "", "");
        println!("{:-<76}", "");

        let mut count = 1;
        for (input, precision, scale) in &test_cases {
            let big_decimal = bigdecimal::BigDecimal::from_str(input).unwrap();
            let (bigint, exponent) = big_decimal.as_bigint_and_exponent();
            let digits = big_decimal.digits();

            assert_eq!(
                *precision as i64, digits as i64,
                "precision != {digits} for {input} | exp={exponent}"
            );
            assert_eq!(*scale as i64, exponent, "scale != {exponent} for {input}");

            // Table row
            println!(
                "{count:>3} | {input:>10}| {big_decimal:>20}| {bigint:>5}| {precision:>5}| {scale:>5}| {:>5}|", big_decimal.normalized()
            );
            count += 1;
        }
        println!("{:-<76}", "");
    }
}
