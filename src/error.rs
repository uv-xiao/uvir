use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Parse error: {0}")]
    ParseError(String),
    
    #[error("Verification error: {0}")]
    VerificationError(String),
    
    #[error("Type error: {0}")]
    TypeError(String),
    
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    #[error("Not found: {0}")]
    NotFound(String),
    
    #[error("Internal error: {0}")]
    InternalError(String),
}

pub type Result<T> = std::result::Result<T, Error>;