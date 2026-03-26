/// Input validation utilities.

/// Maximum message length (128 KB).
pub const MAX_MESSAGE_LEN: usize = 128 * 1024;

/// Maximum tool output length (512 KB).
pub const MAX_TOOL_OUTPUT_LEN: usize = 512 * 1024;

#[derive(Debug)]
pub enum ValidationError {
    TooLong { len: usize, max: usize },
    Empty,
    NullByte(usize),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooLong { len, max } => {
                write!(f, "input too long: {len} bytes (max {max})")
            }
            Self::Empty => write!(f, "input is empty"),
            Self::NullByte(pos) => write!(f, "null byte at position {pos}"),
        }
    }
}

/// Validate user input.
pub fn validate_input(text: &str) -> Result<(), ValidationError> {
    if text.is_empty() {
        return Err(ValidationError::Empty);
    }
    if text.len() > MAX_MESSAGE_LEN {
        return Err(ValidationError::TooLong {
            len: text.len(),
            max: MAX_MESSAGE_LEN,
        });
    }
    if let Some(pos) = text.bytes().position(|b| b == 0) {
        return Err(ValidationError::NullByte(pos));
    }
    Ok(())
}

/// Validate tool output.
pub fn validate_tool_output(text: &str) -> Result<(), ValidationError> {
    if text.len() > MAX_TOOL_OUTPUT_LEN {
        return Err(ValidationError::TooLong {
            len: text.len(),
            max: MAX_TOOL_OUTPUT_LEN,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_input() {
        assert!(validate_input("hello world").is_ok());
    }

    #[test]
    fn test_empty_input() {
        assert!(matches!(validate_input(""), Err(ValidationError::Empty)));
    }

    #[test]
    fn test_too_long() {
        let long = "x".repeat(MAX_MESSAGE_LEN + 1);
        assert!(matches!(
            validate_input(&long),
            Err(ValidationError::TooLong { .. })
        ));
    }

    #[test]
    fn test_null_byte() {
        assert!(matches!(
            validate_input("hello\0world"),
            Err(ValidationError::NullByte(5))
        ));
    }
}
