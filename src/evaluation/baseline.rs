//! Transparent baselines for comparing an advisory learned model.

use super::{ClassificationMetrics, EvaluationError, evaluate_binary_classification};
use crate::linalg::Vector;

/// Majority-class baseline fitted from binary training labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MajorityClassBaseline {
    label: u8,
}

impl MajorityClassBaseline {
    /// Fits the most common label. Ties intentionally select `0` (safe) so
    /// behaviour is stable and visible rather than accidental.
    pub fn fit(labels: &[f64]) -> Result<Self, EvaluationError> {
        let metrics = evaluate_binary_classification(labels, labels)?;
        let positives = metrics.confusion_matrix.true_positive();
        let label = u8::from(positives * 2 > labels.len());
        Ok(Self { label })
    }

    /// Returns the constant class selected from training data.
    pub fn label(&self) -> u8 {
        self.label
    }

    /// Predicts the fitted class for each supplied sample.
    pub fn predict(&self, sample_count: usize) -> Vec<f64> {
        vec![self.label as f64; sample_count]
    }
}

/// Deterministic transparent heuristic: a positive breaking-change count is breaking.
pub fn breaking_change_heuristic(features: &[Vector]) -> Result<Vec<f64>, EvaluationError> {
    features
        .iter()
        .map(|features| {
            if features.len() <= 6 {
                return Err(EvaluationError::FeatureDimensionMismatch {
                    expected_at_least: 7,
                    actual: features.len(),
                });
            }
            let count = features[6];
            if !count.is_finite() || count < 0.0 {
                return Err(EvaluationError::InvalidScore { value: count });
            }
            Ok(f64::from(count > 0.0))
        })
        .collect()
}

/// Evaluates a named transparent baseline with the shared binary metrics.
pub fn evaluate_baseline(
    predictions: &[f64],
    targets: &[f64],
) -> Result<ClassificationMetrics, EvaluationError> {
    evaluate_binary_classification(predictions, targets)
}

/// Side-by-side evaluation of transparent baselines and an advisory neural model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaselineComparisonReport {
    /// Constant prediction fitted from training labels only.
    pub majority: ClassificationMetrics,
    /// Deterministic `breaking_change_count > 0` heuristic.
    pub heuristic: ClassificationMetrics,
    /// Metrics from the learned model's thresholded labels.
    pub neural_model: ClassificationMetrics,
    /// Optional authoritative deterministic policy comparison when labels are available.
    pub deterministic_policy: Option<ClassificationMetrics>,
}

/// Compares common transparent baselines with neural labels on one evaluation set.
///
/// `training_labels` must come from the training partition; all other inputs
/// belong to the same isolated validation or test partition.
pub fn compare_classification_baselines(
    training_labels: &[f64],
    evaluation_features: &[Vector],
    evaluation_targets: &[f64],
    neural_predictions: &[f64],
    deterministic_policy_predictions: Option<&[f64]>,
) -> Result<BaselineComparisonReport, EvaluationError> {
    let majority = MajorityClassBaseline::fit(training_labels)?;
    let heuristic_predictions = breaking_change_heuristic(evaluation_features)?;
    Ok(BaselineComparisonReport {
        majority: evaluate_baseline(
            &majority.predict(evaluation_targets.len()),
            evaluation_targets,
        )?,
        heuristic: evaluate_baseline(&heuristic_predictions, evaluation_targets)?,
        neural_model: evaluate_binary_classification(neural_predictions, evaluation_targets)?,
        deterministic_policy: deterministic_policy_predictions
            .map(|predictions| evaluate_binary_classification(predictions, evaluation_targets))
            .transpose()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_majority_and_applies_breaking_count_heuristic() {
        let baseline = MajorityClassBaseline::fit(&[0.0, 1.0, 1.0]).unwrap();
        assert_eq!(baseline.label(), 1);
        assert_eq!(baseline.predict(2), vec![1.0, 1.0]);
        assert_eq!(
            breaking_change_heuristic(&[
                Vector::new(vec![0.0; 8]),
                Vector::new(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]),
            ])
            .unwrap(),
            vec![0.0, 1.0]
        );
    }

    #[test]
    fn compares_baselines_without_claiming_model_superiority() {
        let report = compare_classification_baselines(
            &[0.0, 1.0, 1.0],
            &[
                Vector::new(vec![0.0; 8]),
                Vector::new(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            ],
            &[0.0, 1.0],
            &[0.0, 1.0],
            Some(&[0.0, 1.0]),
        )
        .unwrap();
        assert_eq!(report.heuristic.accuracy, 1.0);
        assert_eq!(report.deterministic_policy.unwrap().recall, 1.0);
    }
}
