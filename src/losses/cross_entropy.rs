use crate::linalg::Vector;

use super::loss::{
    LossError, clamp_probability, validate_distribution, validate_pair, validate_probability,
    validate_target,
};

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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-10, "{left} != {right}");
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
