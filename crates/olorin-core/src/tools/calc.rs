use super::Tool;
use async_trait::async_trait;

pub struct CalcTool;

#[async_trait]
impl Tool for CalcTool {
    fn name(&self) -> &str {
        "calc"
    }

    fn description(&self) -> &str {
        "Evaluate a math expression. Supports +, -, *, /, %, parentheses."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "expr": {
                    "type": "string",
                    "description": "The math expression to evaluate"
                }
            },
            "required": ["expr"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let expr = params["expr"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'expr' parameter".into()))?;
        // Try integer arithmetic first for exact results on whole numbers
        if let Ok(v) = eval_int(expr) {
            return Ok(format!("{v}"));
        }
        match eval(expr) {
            Ok(v) => {
                if v == v.floor() && v.abs() < 1e15 {
                    Ok(format!("{}", v as i64))
                } else {
                    Ok(format!("{v}"))
                }
            }
            Err(e) => Err(crate::error::Error::Tool(format!("calc: {e}"))),
        }
    }
}

// Integer evaluator using i128 for exact arithmetic on whole numbers.
// Returns Err if the expression contains floats or results in non-integer values.
fn eval_int(input: &str) -> std::result::Result<i128, String> {
    let tokens = tokenize(input)?;
    // Reject if any token is a non-integer number
    for t in &tokens {
        if let Token::Num(n) = t {
            if *n != n.floor() || n.abs() > i128::MAX as f64 {
                return Err("not integer".into());
            }
        }
    }
    let mut pos = 0;
    let result = parse_expr_int(&tokens, &mut pos)?;
    if pos < tokens.len() {
        return Err(format!("unexpected token: {:?}", tokens[pos]));
    }
    Ok(result)
}

fn parse_expr_int(tokens: &[Token], pos: &mut usize) -> std::result::Result<i128, String> {
    let mut left = parse_term_int(tokens, pos)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Op('+') => { *pos += 1; left = left.checked_add(parse_term_int(tokens, pos)?).ok_or("overflow")?; }
            Token::Op('-') => { *pos += 1; left = left.checked_sub(parse_term_int(tokens, pos)?).ok_or("overflow")?; }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_term_int(tokens: &[Token], pos: &mut usize) -> std::result::Result<i128, String> {
    let mut left = parse_unary_int(tokens, pos)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Op('*') => { *pos += 1; left = left.checked_mul(parse_unary_int(tokens, pos)?).ok_or("overflow")?; }
            Token::Op('/') => {
                *pos += 1;
                let r = parse_unary_int(tokens, pos)?;
                if r == 0 { return Err("division by zero".into()); }
                if left % r != 0 { return Err("not integer".into()); }
                left /= r;
            }
            Token::Op('%') => {
                *pos += 1;
                let r = parse_unary_int(tokens, pos)?;
                if r == 0 { return Err("modulo by zero".into()); }
                left %= r;
            }
            _ => break,
        }
    }
    Ok(left)
}

fn parse_unary_int(tokens: &[Token], pos: &mut usize) -> std::result::Result<i128, String> {
    if *pos < tokens.len() {
        if let Token::Op('-') = &tokens[*pos] {
            *pos += 1;
            return Ok(-parse_unary_int(tokens, pos)?);
        }
    }
    parse_atom_int(tokens, pos)
}

fn parse_atom_int(tokens: &[Token], pos: &mut usize) -> std::result::Result<i128, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of expression".into());
    }
    match &tokens[*pos] {
        Token::Num(n) => { let v = *n as i128; *pos += 1; Ok(v) }
        Token::LParen => {
            *pos += 1;
            let v = parse_expr_int(tokens, pos)?;
            if *pos >= tokens.len() {
                return Err("missing closing parenthesis".into());
            }
            match &tokens[*pos] {
                Token::RParen => { *pos += 1; Ok(v) }
                _ => Err("expected closing parenthesis".into()),
            }
        }
        t => Err(format!("unexpected token: {t:?}")),
    }
}

// Tiny recursive descent expression evaluator. No deps.
fn eval(input: &str) -> std::result::Result<f64, String> {
    let tokens = tokenize(input)?;
    let mut pos = 0;
    let result = parse_expr(&tokens, &mut pos)?;
    if pos < tokens.len() {
        return Err(format!("unexpected token: {:?}", tokens[pos]));
    }
    Ok(result)
}

#[derive(Debug, Clone)]
enum Token {
    Num(f64),
    Op(char),
    LParen,
    RParen,
}

fn tokenize(input: &str) -> std::result::Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' => i += 1,
            b'(' => { tokens.push(Token::LParen); i += 1; }
            b')' => { tokens.push(Token::RParen); i += 1; }
            b'+' | b'-' | b'*' | b'/' | b'%' => {
                tokens.push(Token::Op(bytes[i] as char));
                i += 1;
            }
            b'0'..=b'9' | b'.' => {
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                let s = &input[start..i];
                let n: f64 = s.parse().map_err(|_| format!("invalid number: {s}"))?;
                tokens.push(Token::Num(n));
            }
            c => return Err(format!("unexpected character: {}", c as char)),
        }
    }
    Ok(tokens)
}

// expr = term (('+' | '-') term)*
fn parse_expr(tokens: &[Token], pos: &mut usize) -> std::result::Result<f64, String> {
    let mut left = parse_term(tokens, pos)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Op('+') => { *pos += 1; left += parse_term(tokens, pos)?; }
            Token::Op('-') => { *pos += 1; left -= parse_term(tokens, pos)?; }
            _ => break,
        }
    }
    Ok(left)
}

// term = unary (('*' | '/' | '%') unary)*
fn parse_term(tokens: &[Token], pos: &mut usize) -> std::result::Result<f64, String> {
    let mut left = parse_unary(tokens, pos)?;
    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Op('*') => { *pos += 1; left *= parse_unary(tokens, pos)?; }
            Token::Op('/') => {
                *pos += 1;
                let r = parse_unary(tokens, pos)?;
                if r == 0.0 { return Err("division by zero".into()); }
                left /= r;
            }
            Token::Op('%') => {
                *pos += 1;
                let r = parse_unary(tokens, pos)?;
                if r == 0.0 { return Err("modulo by zero".into()); }
                left %= r;
            }
            _ => break,
        }
    }
    Ok(left)
}

// unary = '-' unary | atom
fn parse_unary(tokens: &[Token], pos: &mut usize) -> std::result::Result<f64, String> {
    if *pos < tokens.len() {
        if let Token::Op('-') = &tokens[*pos] {
            *pos += 1;
            return Ok(-parse_unary(tokens, pos)?);
        }
    }
    parse_atom(tokens, pos)
}

// atom = number | '(' expr ')'
fn parse_atom(tokens: &[Token], pos: &mut usize) -> std::result::Result<f64, String> {
    if *pos >= tokens.len() {
        return Err("unexpected end of expression".into());
    }
    match &tokens[*pos] {
        Token::Num(n) => { let v = *n; *pos += 1; Ok(v) }
        Token::LParen => {
            *pos += 1;
            let v = parse_expr(tokens, pos)?;
            if *pos >= tokens.len() {
                return Err("missing closing parenthesis".into());
            }
            match &tokens[*pos] {
                Token::RParen => { *pos += 1; Ok(v) }
                _ => Err("expected closing parenthesis".into()),
            }
        }
        t => Err(format!("unexpected token: {t:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        assert_eq!(eval("2 + 3").unwrap(), 5.0);
        assert_eq!(eval("10 - 4").unwrap(), 6.0);
        assert_eq!(eval("3 * 7").unwrap(), 21.0);
        assert_eq!(eval("20 / 4").unwrap(), 5.0);
        assert_eq!(eval("10 % 3").unwrap(), 1.0);
    }

    #[test]
    fn test_precedence() {
        assert_eq!(eval("2 + 3 * 4").unwrap(), 14.0);
        assert_eq!(eval("(2 + 3) * 4").unwrap(), 20.0);
    }

    #[test]
    fn test_negative() {
        assert_eq!(eval("-5").unwrap(), -5.0);
        assert_eq!(eval("3 * -2").unwrap(), -6.0);
    }

    #[test]
    fn test_float() {
        assert_eq!(eval("1.5 + 2.5").unwrap(), 4.0);
    }

    #[test]
    fn test_div_zero() {
        assert!(eval("1 / 0").is_err());
    }

    #[test]
    fn test_complex() {
        assert_eq!(eval("5673 * 4").unwrap(), 22692.0);
    }
}
