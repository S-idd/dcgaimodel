use crate::nn::NetworkError;
use std::error::Error;
use std::fmt;

/// Errors that can occur while configuring or running an optimizer.
#[derive(Debug, PartialEq)]
pub enum OptimizerError {
    /// Learning rate must be positive and finite.
    InvalidLearningRate { value: f64 },

    /// Momentum must be finite and in the range `0..1`.
    InvalidMomentum { value: f64 },

    /// Adam beta values must be finite and in the range `0..1`.
    InvalidBeta { value: f64 },

    /// Epsilon must be positive and finite.
    InvalidEpsilon { value: f64 },

    /// A network operation failed while applying gradients.
    Network(NetworkError),
}

impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerError::InvalidLearningRate { value } => {
                write!(f, "Learning rate must be positive and finite: {}", value)
            }
            OptimizerError::InvalidMomentum { value } => {
                write!(
                    f,
                    "Momentum must be finite and in the range 0..1: {}",
                    value
                )
            }
            OptimizerError::InvalidBeta { value } => {
                write!(f, "Beta must be finite and in the range 0..1: {}", value)
            }
            OptimizerError::InvalidEpsilon { value } => {
                write!(f, "Epsilon must be positive and finite: {}", value)
            }
            OptimizerError::Network(error) => write!(f, "Network optimizer error: {}", error),
        }
    }
}

impl Error for OptimizerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            OptimizerError::Network(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NetworkError> for OptimizerError {
    fn from(error: NetworkError) -> Self {
        OptimizerError::Network(error)
    }
}
