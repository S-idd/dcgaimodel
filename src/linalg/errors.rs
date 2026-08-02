use std::fmt;

/// Errors that can occur during vector operations.
#[derive(Debug, PartialEq)]
pub enum VectorError {
    /// The vectors do not have the same length.
    DimensionMismatch {
        left: usize,
        right: usize,
    },

    /// Tried to access an invalid index.
    IndexOutOfBounds {
        index: usize,
        length: usize,
    },

    ZeroMagnitude,
}

impl fmt::Display for VectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorError::DimensionMismatch { left, right } => {
                write!(f, "Dimension mismatch: left = {}, right = {}", left, right)
            }

            VectorError::IndexOutOfBounds { index, length } => {
                write!(
                    f,
                    "Index {} is out of bounds for vector of length {}",
                    index, length
                )
            }

            VectorError::ZeroMagnitude => {
                write!(f, "Cannot normalize a vector with zero magnitude.")
            }
        }
    }
}

impl std::error::Error for VectorError {}
