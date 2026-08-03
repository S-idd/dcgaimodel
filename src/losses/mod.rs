use crate::linalg::Vector;
use std::fmt;

const EPSILON: f64 = 1e-15;
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-10;

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

/// Computes mean squared error between predicted and target values.
pub fn mean_squared_error(predicted: &Vector, target: &Vector) -> Result<f64, LossError> {
    validate_pair(predicted, target)?;

    let sum = predicted
        .iter()
        .zip(target.iter())
        .map(|(prediction, expected)| {
            let error = prediction - expected;
            error * error
        })
        .sum::<f64>();

    Ok(sum / predicted.len() as f64)
}

/// Computes binary cross entropy between predicted probabilities and targets.
///
/// Predicted probabilities and targets must be in the inclusive range `0..=1`.
/// Probabilities are clamped internally before taking logarithms so exact 0 and
/// 1 inputs remain finite.
pub fn binary_cross_entropy(predicted: &Vector, target: &Vector) -> Result<f64, LossError> {
    validate_pair(predicted, target)?;
    validate_probabilities(predicted)?;
    validate_targets(target)?;

    let sum = predicted
        .iter()
        .zip(target.iter())
        .map(|(prediction, expected)| {
            let probability = clamp_probability(*prediction);
            -(expected * probability.ln() + (1.0 - expected) * (1.0 - probability).ln())
        })
        .sum::<f64>();

    Ok(sum / predicted.len() as f64)
}

/// Computes multiclass cross entropy between two probability distributions.
///
/// Both vectors must contain probabilities in the inclusive range `0..=1` and
/// each vector must sum to 1.
pub fn cross_entropy(predicted: &Vector, target: &Vector) -> Result<f64, LossError> {
    validate_pair(predicted, target)?;
    validate_distribution(predicted, validate_probability)?;
    validate_distribution(target, validate_target)?;

    let sum = predicted
        .iter()
        .zip(target.iter())
        .map(|(prediction, expected)| -expected * clamp_probability(*prediction).ln())
        .sum::<f64>();

    Ok(sum)
}

fn validate_pair(predicted: &Vector, target: &Vector) -> Result<(), LossError> {
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

fn validate_probabilities(values: &Vector) -> Result<(), LossError> {
    for value in values {
        validate_probability(*value)?;
    }

    Ok(())
}

fn validate_targets(values: &Vector) -> Result<(), LossError> {
    for value in values {
        validate_target(*value)?;
    }

    Ok(())
}

fn validate_probability(value: f64) -> Result<(), LossError> {
    if !is_probability(value) {
        return Err(LossError::InvalidProbability { value });
    }

    Ok(())
}

fn validate_target(value: f64) -> Result<(), LossError> {
    if !is_probability(value) {
        return Err(LossError::InvalidTarget { value });
    }

    Ok(())
}

fn validate_distribution<F>(values: &Vector, validate_value: F) -> Result<(), LossError>
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

fn is_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn clamp_probability(value: f64) -> f64 {
    value.clamp(EPSILON, 1.0 - EPSILON)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-10, "{left} != {right}");
    }

    #[test]
    fn mean_squared_error_computes_average_squared_error() {
        let predicted = Vector::new(vec![2.0, 4.0, 6.0]);
        let target = Vector::new(vec![1.0, 2.0, 3.0]);

        let result = mean_squared_error(&predicted, &target).unwrap();

        assert_close(result, 14.0 / 3.0);
    }

    #[test]
    fn mean_squared_error_is_zero_for_equal_vectors() {
        let predicted = Vector::new(vec![1.0, 2.0, 3.0]);
        let target = Vector::new(vec![1.0, 2.0, 3.0]);

        let result = mean_squared_error(&predicted, &target).unwrap();

        assert_eq!(result, 0.0);
    }

    #[test]
    fn mean_squared_error_dimension_mismatch_returns_error() {
        let predicted = Vector::new(vec![1.0, 2.0]);
        let target = Vector::new(vec![1.0]);

        let result = mean_squared_error(&predicted, &target);

        assert_eq!(
            result,
            Err(LossError::DimensionMismatch {
                predicted: 2,
                target: 1
            })
        );
    }

    #[test]
    fn mean_squared_error_empty_input_returns_error() {
        let predicted = Vector::new(vec![]);
        let target = Vector::new(vec![]);

        let result = mean_squared_error(&predicted, &target);

        assert_eq!(result, Err(LossError::EmptyInput));
    }

    #[test]
    fn binary_cross_entropy_computes_average_loss() {
        let predicted = Vector::new(vec![0.9, 0.1]);
        let target = Vector::new(vec![1.0, 0.0]);

        let result = binary_cross_entropy(&predicted, &target).unwrap();
        let expected = -0.9_f64.ln();

        assert_close(result, expected);
    }

    #[test]
    fn binary_cross_entropy_handles_exact_zero_and_one_probabilities() {
        let predicted = Vector::new(vec![1.0, 0.0]);
        let target = Vector::new(vec![1.0, 0.0]);

        let result = binary_cross_entropy(&predicted, &target).unwrap();

        assert!(result.is_finite());
        assert!(result < 1e-12);
    }

    #[test]
    fn binary_cross_entropy_dimension_mismatch_returns_error() {
        let predicted = Vector::new(vec![0.5, 0.5]);
        let target = Vector::new(vec![1.0]);

        let result = binary_cross_entropy(&predicted, &target);

        assert_eq!(
            result,
            Err(LossError::DimensionMismatch {
                predicted: 2,
                target: 1
            })
        );
    }

    #[test]
    fn binary_cross_entropy_rejects_invalid_probability() {
        let predicted = Vector::new(vec![1.2]);
        let target = Vector::new(vec![1.0]);

        let result = binary_cross_entropy(&predicted, &target);

        assert_eq!(result, Err(LossError::InvalidProbability { value: 1.2 }));
    }

    #[test]
    fn binary_cross_entropy_rejects_invalid_target() {
        let predicted = Vector::new(vec![0.8]);
        let target = Vector::new(vec![-1.0]);

        let result = binary_cross_entropy(&predicted, &target);

        assert_eq!(result, Err(LossError::InvalidTarget { value: -1.0 }));
    }

    #[test]
    fn cross_entropy_computes_multiclass_loss() {
        let predicted = Vector::new(vec![0.1, 0.7, 0.2]);
        let target = Vector::new(vec![0.0, 1.0, 0.0]);

        let result = cross_entropy(&predicted, &target).unwrap();

        assert_close(result, -0.7_f64.ln());
    }

    #[test]
    fn cross_entropy_supports_soft_targets() {
        let predicted = Vector::new(vec![0.2, 0.5, 0.3]);
        let target = Vector::new(vec![0.1, 0.7, 0.2]);

        let result = cross_entropy(&predicted, &target).unwrap();
        let expected = -(0.1 * 0.2_f64.ln() + 0.7 * 0.5_f64.ln() + 0.2 * 0.3_f64.ln());

        assert_close(result, expected);
    }

    #[test]
    fn cross_entropy_handles_zero_predicted_probability() {
        let predicted = Vector::new(vec![0.0, 1.0]);
        let target = Vector::new(vec![1.0, 0.0]);

        let result = cross_entropy(&predicted, &target).unwrap();

        assert!(result.is_finite());
        assert!(result > 30.0);
    }

    #[test]
    fn cross_entropy_dimension_mismatch_returns_error() {
        let predicted = Vector::new(vec![0.5, 0.5]);
        let target = Vector::new(vec![1.0]);

        let result = cross_entropy(&predicted, &target);

        assert_eq!(
            result,
            Err(LossError::DimensionMismatch {
                predicted: 2,
                target: 1
            })
        );
    }

    #[test]
    fn cross_entropy_rejects_invalid_probability() {
        let predicted = Vector::new(vec![1.1, -0.1]);
        let target = Vector::new(vec![1.0, 0.0]);

        let result = cross_entropy(&predicted, &target);

        assert_eq!(result, Err(LossError::InvalidProbability { value: 1.1 }));
    }

    #[test]
    fn cross_entropy_rejects_invalid_target() {
        let predicted = Vector::new(vec![0.5, 0.5]);
        let target = Vector::new(vec![1.2, -0.2]);

        let result = cross_entropy(&predicted, &target);

        assert_eq!(result, Err(LossError::InvalidTarget { value: 1.2 }));
    }

    #[test]
    fn cross_entropy_rejects_prediction_distribution_that_does_not_sum_to_one() {
        let predicted = Vector::new(vec![0.4, 0.4]);
        let target = Vector::new(vec![1.0, 0.0]);

        let result = cross_entropy(&predicted, &target);

        assert_eq!(result, Err(LossError::InvalidDistribution { sum: 0.8 }));
    }

    #[test]
    fn cross_entropy_rejects_target_distribution_that_does_not_sum_to_one() {
        let predicted = Vector::new(vec![0.5, 0.5]);
        let target = Vector::new(vec![0.5, 0.3]);

        let result = cross_entropy(&predicted, &target);

        assert_eq!(result, Err(LossError::InvalidDistribution { sum: 0.8 }));
    }
}
