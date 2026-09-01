use super::{ClassificationMetrics, EvaluationError, evaluate_binary_classification};

/// Classification metrics produced by applying one explicit threshold to scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThresholdEvaluation {
    /// The threshold used to map scores to binary labels.
    pub threshold: f64,
    /// Resulting binary classification metrics.
    pub metrics: ClassificationMetrics,
}

/// Evaluates continuous bounded scores at a caller-selected classification threshold.
pub fn evaluate_threshold(
    scores: &[f64],
    targets: &[f64],
    threshold: f64,
) -> Result<ThresholdEvaluation, EvaluationError> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(EvaluationError::InvalidThreshold { value: threshold });
    }
    let labels = scores
        .iter()
        .map(|score| {
            if !score.is_finite() || !(0.0..=1.0).contains(score) {
                Err(EvaluationError::InvalidScore { value: *score })
            } else {
                Ok(f64::from(*score >= threshold))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ThresholdEvaluation {
        threshold,
        metrics: evaluate_binary_classification(&labels, targets)?,
    })
}

/// Evaluates an arbitrary list of valid thresholds without selecting a production value.
pub fn evaluate_thresholds(
    scores: &[f64],
    targets: &[f64],
    thresholds: &[f64],
) -> Result<Vec<ThresholdEvaluation>, EvaluationError> {
    thresholds
        .iter()
        .map(|threshold| evaluate_threshold(scores, targets, *threshold))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_threshold_false_positive_and_false_negative_tradeoffs() {
        let low = evaluate_threshold(&[0.4, 0.6], &[0.0, 1.0], 0.3).unwrap();
        let high = evaluate_threshold(&[0.4, 0.6], &[0.0, 1.0], 0.7).unwrap();
        assert_eq!(low.metrics.confusion_matrix.false_positive(), 1);
        assert_eq!(low.metrics.confusion_matrix.false_negative(), 0);
        assert_eq!(high.metrics.confusion_matrix.false_positive(), 0);
        assert_eq!(high.metrics.confusion_matrix.false_negative(), 1);
    }

    #[test]
    fn rejects_invalid_thresholds_and_scores() {
        assert!(matches!(
            evaluate_threshold(&[0.5], &[1.0], f64::NAN),
            Err(EvaluationError::InvalidThreshold { .. })
        ));
        assert_eq!(
            evaluate_threshold(&[1.1], &[1.0], 0.5),
            Err(EvaluationError::InvalidScore { value: 1.1 })
        );
    }
}
