use std::fmt;

pub enum Error {
    Io(std::io::Error),
    Kernel(&'static str),
    Vault(&'static str),
    Inference(&'static str),
    Safety(String),
    Json(&'static str),
    Config(String),
    Tool(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Kernel(msg) => write!(f, "kernel error: {msg}"),
            Error::Vault(msg) => write!(f, "vault error: {msg}"),
            Error::Inference(msg) => write!(f, "inference error: {msg}"),
            Error::Safety(msg) => write!(f, "safety violation: {msg}"),
            Error::Json(msg) => write!(f, "JSON error: {msg}"),
            Error::Config(msg) => write!(f, "config error: {msg}"),
            Error::Tool(msg) => write!(f, "tool error: {msg}"),
        }
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
