use crate::linalg::Vector;

use super::loss::{Loss, LossError, validate_pair};

#[derive(Debug, Clone, Copy, Default)]
pub struct MeanSquaredError;
impl MeanSquaredError {
    /// Creates a new Mean Squared Error loss.
    pub fn new() -> Self {
        Self
    }
}

impl Loss for MeanSquaredError {
    /// Computes the mean squared error between predictions and targets.
    fn forward(&self, predicted: &Vector, target: &Vector) -> Result<f64, LossError> {
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

    /// Computes the gradient of MSE with respect to predictions.
    ///
    /// d(MSE) / d(prediction) = 2 * (prediction - target) / n
    fn backward(&self, predicted: &Vector, target: &Vector) -> Result<Vector, LossError> {
        validate_pair(predicted, target)?;

        let scale = 2.0 / predicted.len() as f64;

        let gradient = predicted
            .iter()
            .zip(target.iter())
            .map(|(prediction, expected)| scale * (prediction - expected))
            .collect();

        Ok(Vector::new(gradient))
    }
}

/// Computes mean squared error between predicted and target values.
///
/// This function is kept as a convenient functional API.
pub fn mean_squared_error(predicted: &Vector, target: &Vector) -> Result<f64, LossError> {
    MeanSquaredError::new().forward(predicted, target)
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
    fn mse_forward_matches_functional_api() {
        let predicted = Vector::new(vec![2.0, 4.0, 6.0]);
        let target = Vector::new(vec![1.0, 2.0, 3.0]);

        let loss = MeanSquaredError::new();

        let result = loss.forward(&predicted, &target).unwrap();
        let expected = mean_squared_error(&predicted, &target).unwrap();

        assert_close(result, expected);
    }

    #[test]
    fn mse_backward_computes_gradient() {
        let predicted = Vector::new(vec![2.0, 4.0, 6.0]);
        let target = Vector::new(vec![1.0, 2.0, 3.0]);

        let loss = MeanSquaredError::new();

        let gradient = loss.backward(&predicted, &target).unwrap();

        assert_close(gradient[0], 2.0 / 3.0);
        assert_close(gradient[1], 4.0 / 3.0);
        assert_close(gradient[2], 2.0);
    }

    #[test]
    fn mse_backward_is_zero_for_equal_vectors() {
        let predicted = Vector::new(vec![1.0, 2.0, 3.0]);
        let target = Vector::new(vec![1.0, 2.0, 3.0]);

        let loss = MeanSquaredError::new();

        let gradient = loss.backward(&predicted, &target).unwrap();

        assert_eq!(gradient, Vector::new(vec![0.0, 0.0, 0.0]));
    }

    #[test]
    fn mse_backward_rejects_dimension_mismatch() {
        let predicted = Vector::new(vec![1.0, 2.0]);
        let target = Vector::new(vec![1.0]);

        let loss = MeanSquaredError::new();

        let result = loss.backward(&predicted, &target);

        assert_eq!(
            result,
            Err(LossError::DimensionMismatch {
                predicted: 2,
                target: 1
            })
        );
    }

    #[test]
    fn mse_backward_rejects_empty_input() {
        let predicted = Vector::new(vec![]);
        let target = Vector::new(vec![]);

        let loss = MeanSquaredError::new();

        let result = loss.backward(&predicted, &target);

        assert_eq!(result, Err(LossError::EmptyInput));
    }
}
