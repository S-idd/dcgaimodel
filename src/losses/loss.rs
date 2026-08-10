use crate::linalg::Vector;
use std::fmt;

pub(super) const EPSILON: f64 = 1e-15;
pub(super) const PROBABILITY_SUM_TOLERANCE: f64 = 1e-10;

/// Errors that can occur while computing loss functions.
#[derive(Debug, Clone, PartialEq)]
pub enum LossError {
    /// The prediction and target vectors do not have the same length.
    DimensionMismatch { predicted: usize, target: usize },

    /// A loss was requested for an empty vector.
    EmptyInput,

    /// A predicted probability was outside the inclusive range `0..=1`.
    InvalidProbability { value: f64 },

    /// A target value was outside the inclusive range `0..=1`.
    InvalidTarget { value: f64 },

    /// A probability distribution did not sum to 1.
    InvalidDistribution { sum: f64 },
}

impl fmt::Display for LossError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LossError::DimensionMismatch { predicted, target } => write!(
                f,
                "Dimension mismatch: predicted = {}, target = {}",
                predicted, target
            ),
            LossError::EmptyInput => write!(f, "Cannot compute loss for empty input."),
            LossError::InvalidProbability { value } => {
                write!(f, "Invalid predicted probability: {}", value)
            }
            LossError::InvalidTarget { value } => write!(f, "Invalid target value: {}", value),
            LossError::InvalidDistribution { sum } => {
                write!(f, "Invalid probability distribution sum: {}", sum)
            }
        }
    }
}

impl std::error::Error for LossError {}

/// Common interface implemented by all loss functions.
pub trait Loss {
    /// Computes the loss between predictions and targets.
    fn forward(&self, predicted: &Vector, target: &Vector) -> Result<f64, LossError>;

    /// Computes the gradient of the loss with respect to predictions.
    fn backward(&self, predicted: &Vector, target: &Vector) -> Result<Vector, LossError>;
}

/// Validates that two vectors share the same, non-zero length.
///
/// Shared by every loss function since each one operates on a
/// predicted/target pair of equal size.
pub(super) fn validate_pair(predicted: &Vector, target: &Vector) -> Result<(), LossError> {
    if predicted.len() != target.len() {
        return Err(LossError::DimensionMismatch {
            predicted: predicted.len(),
            target: target.len(),
        });
    }

    if predicted.is_empty() {
        return Err(LossError::EmptyInput);
    }

    Ok(())
}

pub(super) fn validate_probabilities(values: &Vector) -> Result<(), LossError> {
    for value in values {
        validate_probability(*value)?;
    }

    Ok(())
}

pub(super) fn validate_targets(values: &Vector) -> Result<(), LossError> {
    for value in values {
        validate_target(*value)?;
    }

    Ok(())
}

pub(super) fn validate_probability(value: f64) -> Result<(), LossError> {
    if !is_probability(value) {
        return Err(LossError::InvalidProbability { value });
    }

    Ok(())
}

pub(super) fn validate_target(value: f64) -> Result<(), LossError> {
    if !is_probability(value) {
        return Err(LossError::InvalidTarget { value });
    }

    Ok(())
}

pub(super) fn validate_distribution<F>(values: &Vector, validate_value: F) -> Result<(), LossError>
where
    F: Fn(f64) -> Result<(), LossError>,
{
    for value in values {
        validate_value(*value)?;
    }

    let sum = values.iter().sum::<f64>();

    if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        return Err(LossError::InvalidDistribution { sum });
    }

    Ok(())
}

pub(super) fn is_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

pub(super) fn clamp_probability(value: f64) -> f64 {
    value.clamp(EPSILON, 1.0 - EPSILON)
}
