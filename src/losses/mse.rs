use crate::linalg::Vector;

use super::loss::{LossError, validate_pair};

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
}
