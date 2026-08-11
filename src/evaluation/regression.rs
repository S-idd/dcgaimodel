use super::{EvaluationError, validate_pairs};
use crate::linalg::Vector;
use crate::losses::mean_squared_error;

/// Mean squared and mean absolute error for regression outputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegressionMetrics {
    /// Mean of squared prediction errors.
    pub mean_squared_error: f64,
    /// Mean of absolute prediction errors.
    pub mean_absolute_error: f64,
}

/// Evaluates regression predictions using MSE and MAE.
///
/// MSE is delegated to the existing loss implementation after finite-input
/// validation; MAE is the mean absolute difference.
pub fn evaluate_regression(
    predictions: &[f64],
    targets: &[f64],
) -> Result<RegressionMetrics, EvaluationError> {
    validate_pairs(predictions, targets)?;
    let predicted = Vector::new(predictions.to_vec());
    let target = Vector::new(targets.to_vec());
    let mean_squared_error = mean_squared_error(&predicted, &target)?;
    let mean_absolute_error = predictions
        .iter()
        .zip(targets)
        .map(|(prediction, target)| (prediction - target).abs())
        .sum::<f64>()
        / predictions.len() as f64;
    Ok(RegressionMetrics {
        mean_squared_error,
        mean_absolute_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculates_regression_metrics() {
        let metrics = evaluate_regression(&[1.0, 3.0], &[2.0, 1.0]).unwrap();
        assert_eq!(metrics.mean_squared_error, 2.5);
        assert_eq!(metrics.mean_absolute_error, 1.5);
    }

    #[test]
    fn rejects_invalid_regression_input() {
        assert_eq!(
            evaluate_regression(&[], &[]),
            Err(EvaluationError::EmptyInput)
        );
        assert_eq!(
            evaluate_regression(&[1.0], &[1.0, 2.0]),
            Err(EvaluationError::LengthMismatch {
                predictions: 1,
                targets: 2
            })
        );
        assert!(
            matches!(evaluate_regression(&[f64::NAN], &[1.0]), Err(EvaluationError::NonFiniteValue { value }) if value.is_nan())
        );
    }
}
