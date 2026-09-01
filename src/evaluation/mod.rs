//! Model-evaluation metrics for DCG predictions.

mod baseline;
mod classification;
mod regression;
mod threshold;

pub use baseline::{
    BaselineComparisonReport, MajorityClassBaseline, breaking_change_heuristic,
    compare_classification_baselines, evaluate_baseline,
};
pub use classification::{ClassificationMetrics, ConfusionMatrix, evaluate_binary_classification};
pub use regression::{RegressionMetrics, evaluate_regression};
pub use threshold::{ThresholdEvaluation, evaluate_threshold, evaluate_thresholds};

use std::error::Error;
use std::fmt;

/// Errors returned while validating prediction/target pairs for evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    /// At least one prediction/target pair is required.
    EmptyInput,
    /// Predictions and targets do not have the same number of values.
    LengthMismatch { predictions: usize, targets: usize },
    /// A binary label was not `0` or `1`.
    InvalidLabel { value: f64 },
    /// A value was NaN or infinite.
    NonFiniteValue { value: f64 },
    /// A bounded score was outside `0.0..=1.0`.
    InvalidScore { value: f64 },
    /// A classification threshold was invalid.
    InvalidThreshold { value: f64 },
    /// A feature vector did not contain enough data for an evaluation baseline.
    FeatureDimensionMismatch {
        expected_at_least: usize,
        actual: usize,
    },
    /// Existing loss validation failed.
    Loss(crate::losses::LossError),
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "Evaluation requires at least one value."),
            Self::LengthMismatch {
                predictions,
                targets,
            } => write!(
                f,
                "Prediction/target length mismatch: predictions = {predictions}, targets = {targets}"
            ),
            Self::InvalidLabel { value } => write!(f, "Invalid binary label: {value}"),
            Self::NonFiniteValue { value } => write!(f, "Non-finite evaluation value: {value}"),
            Self::InvalidScore { value } => write!(f, "Invalid bounded score: {value}"),
            Self::InvalidThreshold { value } => write!(f, "Invalid threshold: {value}"),
            Self::FeatureDimensionMismatch {
                expected_at_least,
                actual,
            } => write!(
                f,
                "Feature vector too short for baseline: expected at least {expected_at_least}, got {actual}"
            ),
            Self::Loss(error) => write!(f, "Loss evaluation error: {error}"),
        }
    }
}

impl Error for EvaluationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Loss(error) => Some(error),
            _ => None,
        }
    }
}

impl From<crate::losses::LossError> for EvaluationError {
    fn from(error: crate::losses::LossError) -> Self {
        Self::Loss(error)
    }
}

fn validate_pairs(predictions: &[f64], targets: &[f64]) -> Result<(), EvaluationError> {
    if predictions.is_empty() || targets.is_empty() {
        return Err(EvaluationError::EmptyInput);
    }
    if predictions.len() != targets.len() {
        return Err(EvaluationError::LengthMismatch {
            predictions: predictions.len(),
            targets: targets.len(),
        });
    }
    for value in predictions.iter().chain(targets) {
        if !value.is_finite() {
            return Err(EvaluationError::NonFiniteValue { value: *value });
        }
    }
    Ok(())
}
