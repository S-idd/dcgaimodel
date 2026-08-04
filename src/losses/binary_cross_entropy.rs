use crate::linalg::Vector;

use super::loss::{
    LossError, clamp_probability, validate_pair, validate_probabilities, validate_targets,
};

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
}
