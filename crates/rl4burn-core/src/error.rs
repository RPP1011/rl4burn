//! Structured error types for rl4burn.

/// Errors that can occur in rl4burn operations.
#[derive(Debug, thiserror::Error)]
pub enum Rl4BurnError {
    /// Environment error with a descriptive message.
    #[error("Environment error: {message}")]
    Environment {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Shape mismatch between expected and actual tensor/observation shapes.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Error during checkpoint save/load.
    #[error("Checkpoint error: {0}")]
    Checkpoint(#[from] std::io::Error),

    /// Configuration error.
    #[error("Config error: {0}")]
    Config(String),
}

/// Convenience result type for rl4burn operations.
pub type Result<T> = std::result::Result<T, Rl4BurnError>;
