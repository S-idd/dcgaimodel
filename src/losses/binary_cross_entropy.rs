use crate::linalg::Vector;

use super::loss::{
    Loss, LossError, clamp_probability, validate_pair, validate_probabilities, validate_targets,
};

/// Binary Cross Entropy loss.
#[derive(Debug, Clone, Copy, Default)]
pub struct BinaryCrossEntropy;

impl BinaryCrossEntropy {
    pub fn new() -> Self {
        Self
    }
}

impl Loss for BinaryCrossEntropy {
    fn forward(&self, predicted: &Vector, target: &Vector) -> Result<f64, LossError> {
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

    fn backward(&self, predicted: &Vector, target: &Vector) -> Result<Vector, LossError> {
        validate_pair(predicted, target)?;
        validate_probabilities(predicted)?;
        validate_targets(target)?;

        let scale = 1.0 / predicted.len() as f64;

        let gradient = predicted
            .iter()
            .zip(target.iter())
            .map(|(prediction, expected)| {
                let probability = clamp_probability(*prediction);

                scale * (probability - expected) / (probability * (1.0 - probability))
            })
            .collect();

        Ok(Vector::new(gradient))
    }
}

/// Computes binary cross entropy between predicted probabilities and targets.
///
/// Predicted probabilities and targets must be in the inclusive range `0..=1`.
/// Probabilities are clamped internally before taking logarithms so exact 0 and
/// 1 inputs remain finite.
/// Computes binary cross entropy between predicted probabilities and targets.
pub fn binary_cross_entropy(predicted: &Vector, target: &Vector) -> Result<f64, LossError> {
    BinaryCrossEntropy::new().forward(predicted, target)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-10, "{left} != {right}");
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
    fn binary_cross_entropy_backward_computes_gradient() {
        let predicted = Vector::new(vec![0.8]);
        let target = Vector::new(vec![1.0]);

        let loss = BinaryCrossEntropy::new();

        let gradient = loss.backward(&predicted, &target).unwrap();

        let expected = (0.8 - 1.0) / (0.8 * 0.2);

        assert_close(gradient[0], expected);
    }

    #[test]
    fn binary_cross_entropy_backward_averages_gradient() {
        let predicted = Vector::new(vec![0.8, 0.2]);
        let target = Vector::new(vec![1.0, 0.0]);

        let loss = BinaryCrossEntropy::new();

        let gradient = loss.backward(&predicted, &target).unwrap();

        let expected_first = ((0.8 - 1.0) / (0.8 * 0.2)) / 2.0;
        let expected_second = ((0.2 - 0.0) / (0.2 * 0.8)) / 2.0;

        assert_close(gradient[0], expected_first);
        assert_close(gradient[1], expected_second);
    }

    #[test]
    fn binary_cross_entropy_backward_rejects_dimension_mismatch() {
        let predicted = Vector::new(vec![0.8, 0.2]);
        let target = Vector::new(vec![1.0]);

        let loss = BinaryCrossEntropy::new();

        let result = loss.backward(&predicted, &target);

        assert_eq!(
            result,
            Err(LossError::DimensionMismatch {
                predicted: 2,
                target: 1
            })
        );
    }
}
