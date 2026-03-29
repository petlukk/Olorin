//! Minimal RFC 8259 JSON parser and serializer.
//!
//! No `serde_json`. Recursive descent. Supports objects, arrays, strings
//! (with escape sequences), integers, floats, booleans, and null.

// ── Public types ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum Value {
    Str(String),
    I64(i64),
    F64(f64),
    Bool(bool),
    Null,
    Array(Vec<Value>),
    Object(Box<Object>),
}

#[derive(Clone, Debug)]
pub struct Object {
    keys:   Vec<String>,
    values: Vec<Value>,
}

impl Object {
    pub fn new() -> Self {
        Self { keys: Vec::new(), values: Vec::new() }
    }

    pub fn set(&mut self, key: &str, value: Value) {
        if let Some(i) = self.keys.iter().position(|k| k == key) {
            self.values[i] = value;
        } else {
            self.keys.push(key.to_owned());
            self.values.push(value);
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.keys.iter().position(|k| k == key)
            .map(|i| &self.values[i])
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.get(key) {
            Some(Value::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_i64(&self, key: &str) -> Option<i64> {
        match self.get(key) {
            Some(Value::I64(n)) => Some(*n),
            _ => None,
        }
    }

    pub fn get_f64(&self, key: &str) -> Option<f64> {
        match self.get(key) {
            Some(Value::F64(f)) => Some(*f),
            Some(Value::I64(n)) => Some(*n as f64),
            _ => None,
        }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.get(key) {
            Some(Value::Bool(b)) => Some(*b),
            _ => None,
        }
    }

    pub fn get_array(&self, key: &str) -> Option<&Vec<Value>> {
        match self.get(key) {
            Some(Value::Array(a)) => Some(a),
            _ => None,
        }
    }

    pub fn get_object(&self, key: &str) -> Option<&Object> {
        match self.get(key) {
            Some(Value::Object(o)) => Some(o),
            _ => None,
        }
    }

    pub fn is_null(&self, key: &str) -> bool {
        matches!(self.get(key), Some(Value::Null))
    }
}

impl Default for Object {
    fn default() -> Self { Self::new() }
}

// ── Parser ────────────────────────────────────────────────────────────────────

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(src: &'a [u8]) -> Self {
        Self { src, pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied();
        if b.is_some() { self.pos += 1; }
        b
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), &'static str> {
        self.skip_ws();
        match self.advance() {
            Some(c) if c == b => Ok(()),
            _ => Err("unexpected character"),
        }
    }

    fn parse_object(&mut self) -> Result<Object, &'static str> {
        self.expect(b'{')?;
        let mut obj = Object::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(obj);
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.expect(b':')?;
            let val = self.parse_value()?;
            obj.set(&key, val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => { self.pos += 1; }
                Some(b'}') => { self.pos += 1; break; }
                _ => return Err("expected ',' or '}'"),
            }
        }
        Ok(obj)
    }

    fn parse_array(&mut self) -> Result<Vec<Value>, &'static str> {
        self.expect(b'[')?;
        let mut arr = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(arr);
        }
        loop {
            let val = self.parse_value()?;
            arr.push(val);
            self.skip_ws();
            match self.peek() {
                Some(b',') => { self.pos += 1; }
                Some(b']') => { self.pos += 1; break; }
                _ => return Err("expected ',' or ']'"),
            }
        }
        Ok(arr)
    }

    fn parse_string(&mut self) -> Result<String, &'static str> {
        self.skip_ws();
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            match self.advance() {
                None => return Err("unterminated string"),
                Some(b'"') => break,
                Some(b'\\') => {
                    match self.advance() {
                        Some(b'"')  => s.push('"'),
                        Some(b'\\') => s.push('\\'),
                        Some(b'/')  => s.push('/'),
                        Some(b'n')  => s.push('\n'),
                        Some(b't')  => s.push('\t'),
                        Some(b'r')  => s.push('\r'),
                        Some(b'b')  => s.push('\x08'),
                        Some(b'f')  => s.push('\x0C'),
                        Some(b'u')  => {
                            let cp = self.parse_hex4()?;
                            // Handle surrogate pairs (UTF-16)
                            let ch = if (0xD800..=0xDBFF).contains(&cp) {
                                // high surrogate — expect \uXXXX low surrogate
                                if self.advance() != Some(b'\\') || self.advance() != Some(b'u') {
                                    return Err("expected low surrogate after high surrogate");
                                }
                                let low = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err("invalid surrogate pair");
                                }
                                let codepoint = 0x10000
                                    + ((cp as u32 - 0xD800) << 10)
                                    + (low as u32 - 0xDC00);
                                char::from_u32(codepoint).ok_or("invalid codepoint")?
                            } else {
                                char::from_u32(cp as u32).ok_or("invalid unicode escape")?
                            };
                            s.push(ch);
                        }
                        _ => return Err("invalid escape sequence"),
                    }
                }
                Some(b) => s.push(b as char),
            }
        }
        Ok(s)
    }

    fn parse_hex4(&mut self) -> Result<u16, &'static str> {
        let mut val = 0u16;
        for _ in 0..4 {
            let b = self.advance().ok_or("unexpected end in unicode escape")?;
            let digit = match b {
                b'0'..=b'9' => (b - b'0') as u16,
                b'a'..=b'f' => (b - b'a') as u16 + 10,
                b'A'..=b'F' => (b - b'A') as u16 + 10,
                _ => return Err("invalid hex digit"),
            };
            val = val * 16 + digit;
        }
        Ok(val)
    }

    fn parse_number(&mut self) -> Result<Value, &'static str> {
        let start = self.pos;
        let mut is_float = false;

        if self.peek() == Some(b'-') { self.pos += 1; }

        // Integer part
        match self.peek() {
            Some(b'0') => { self.pos += 1; }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
            }
            _ => return Err("invalid number"),
        }

        // Fractional part
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err("digit expected after '.'");
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
        }

        // Exponent
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) { self.pos += 1; }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err("digit expected in exponent");
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
        }

        let raw = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| "invalid utf8 in number")?;

        if is_float {
            raw.parse::<f64>().map(Value::F64).map_err(|_| "invalid float")
        } else {
            raw.parse::<i64>().map(Value::I64).map_err(|_| "invalid integer")
        }
    }

    fn parse_value(&mut self) -> Result<Value, &'static str> {
        self.skip_ws();
        match self.peek() {
            Some(b'"') => self.parse_string().map(Value::Str),
            Some(b'{') => self.parse_object().map(|o| Value::Object(Box::new(o))),
            Some(b'[') => self.parse_array().map(Value::Array),
            Some(b't') => {
                self.consume_literal(b"true")?;
                Ok(Value::Bool(true))
            }
            Some(b'f') => {
                self.consume_literal(b"false")?;
                Ok(Value::Bool(false))
            }
            Some(b'n') => {
                self.consume_literal(b"null")?;
                Ok(Value::Null)
            }
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            _ => Err("unexpected token"),
        }
    }

    fn consume_literal(&mut self, lit: &[u8]) -> Result<(), &'static str> {
        for &b in lit {
            match self.advance() {
                Some(c) if c == b => {}
                _ => return Err("invalid literal"),
            }
        }
        Ok(())
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse a JSON object from bytes. Returns `Err(&'static str)` on failure.
pub fn parse(input: &[u8]) -> Result<Object, &'static str> {
    let mut p = Parser::new(input);
    let obj = p.parse_object()?;
    p.skip_ws();
    if p.pos != input.len() {
        return Err("trailing data after JSON object");
    }
    Ok(obj)
}

/// Serialize an Object to a JSON string.
pub fn serialize(obj: &Object) -> String {
    let mut out = String::new();
    write_object(obj, &mut out);
    out
}

/// Serialize any Value to a JSON string.
pub fn serialize_value(val: &Value) -> String {
    let mut out = String::new();
    write_value(val, &mut out);
    out
}

fn write_object(obj: &Object, out: &mut String) {
    out.push('{');
    for (i, key) in obj.keys.iter().enumerate() {
        if i > 0 { out.push(','); }
        write_string(key, out);
        out.push(':');
        write_value(&obj.values[i], out);
    }
    out.push('}');
}

fn write_value(val: &Value, out: &mut String) {
    match val {
        Value::Str(s) => write_string(s, out),
        Value::I64(n) => out.push_str(&n.to_string()),
        Value::F64(f) => {
            // Ensure there is always a decimal point so re-parse gives F64.
            let s = format!("{f}");
            if s.contains('.') || s.contains('e') || s.contains('E') {
                out.push_str(&s);
            } else {
                out.push_str(&s);
                out.push_str(".0");
            }
        }
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Null => out.push_str("null"),
        Value::Array(arr) => {
            out.push('[');
            for (i, v) in arr.iter().enumerate() {
                if i > 0 { out.push(','); }
                write_value(v, out);
            }
            out.push(']');
        }
        Value::Object(o) => write_object(o, out),
    }
}

fn write_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            '\x08' => out.push_str("\\b"),
            '\x0C' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}
