use super::{EvaluationError, validate_pairs};

/// Counts for binary classification, where `0` is negative and `1` is positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ConfusionMatrix {
    true_positive: usize,
    true_negative: usize,
    false_positive: usize,
    false_negative: usize,
}

impl ConfusionMatrix {
    /// Builds a confusion matrix from predicted and actual binary labels.
    ///
    /// Returns an error for empty, mismatched, non-finite, or non-binary input.
    pub fn from_labels(predictions: &[f64], targets: &[f64]) -> Result<Self, EvaluationError> {
        validate_pairs(predictions, targets)?;
        let mut matrix = Self::default();
        for (&prediction, &target) in predictions.iter().zip(targets) {
            if !(prediction == 0.0 || prediction == 1.0) {
                return Err(EvaluationError::InvalidLabel { value: prediction });
            }
            if !(target == 0.0 || target == 1.0) {
                return Err(EvaluationError::InvalidLabel { value: target });
            }
            match (prediction, target) {
                (1.0, 1.0) => matrix.true_positive += 1,
                (0.0, 0.0) => matrix.true_negative += 1,
                (1.0, 0.0) => matrix.false_positive += 1,
                (0.0, 1.0) => matrix.false_negative += 1,
                _ => unreachable!("binary labels were validated"),
            }
        }
        Ok(matrix)
    }

    /// Returns the number of true positives.
    pub fn true_positive(&self) -> usize {
        self.true_positive
    }
    /// Returns the number of true negatives.
    pub fn true_negative(&self) -> usize {
        self.true_negative
    }
    /// Returns the number of false positives.
    pub fn false_positive(&self) -> usize {
        self.false_positive
    }
    /// Returns the number of false negatives.
    pub fn false_negative(&self) -> usize {
        self.false_negative
    }
    /// Returns the total number of evaluated labels.
    pub fn total(&self) -> usize {
        self.true_positive + self.true_negative + self.false_positive + self.false_negative
    }
}

/// Standard binary-classification metrics.
///
/// When a denominator is zero, precision, recall, and F1 deterministically
/// return `0.0`, avoiding NaN and infinity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassificationMetrics {
    /// Fraction of labels predicted correctly.
    pub accuracy: f64,
    /// Fraction of positive predictions that were correct.
    pub precision: f64,
    /// Fraction of actual positives found.
    pub recall: f64,
    /// Harmonic mean of precision and recall.
    pub f1_score: f64,
    /// Underlying binary outcome counts.
    pub confusion_matrix: ConfusionMatrix,
}

/// Evaluates binary labels and computes accuracy, precision, recall, and F1.
pub fn evaluate_binary_classification(
    predictions: &[f64],
    targets: &[f64],
) -> Result<ClassificationMetrics, EvaluationError> {
    let confusion_matrix = ConfusionMatrix::from_labels(predictions, targets)?;
    let total = confusion_matrix.total() as f64;
    let accuracy = (confusion_matrix.true_positive + confusion_matrix.true_negative) as f64 / total;
    let precision_denominator = confusion_matrix.true_positive + confusion_matrix.false_positive;
    let recall_denominator = confusion_matrix.true_positive + confusion_matrix.false_negative;
    let precision = if precision_denominator == 0 {
        0.0
    } else {
        confusion_matrix.true_positive as f64 / precision_denominator as f64
    };
    let recall = if recall_denominator == 0 {
        0.0
    } else {
        confusion_matrix.true_positive as f64 / recall_denominator as f64
    };
    let f1_score = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    Ok(ClassificationMetrics {
        accuracy,
        precision,
        recall,
        f1_score,
        confusion_matrix,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculates_all_binary_metrics() {
        let metrics =
            evaluate_binary_classification(&[1.0, 0.0, 1.0, 0.0], &[1.0, 0.0, 0.0, 1.0]).unwrap();
        assert_eq!(metrics.confusion_matrix.true_positive(), 1);
        assert_eq!(metrics.confusion_matrix.true_negative(), 1);
        assert_eq!(metrics.confusion_matrix.false_positive(), 1);
        assert_eq!(metrics.confusion_matrix.false_negative(), 1);
        assert_eq!(metrics.accuracy, 0.5);
        assert_eq!(metrics.precision, 0.5);
        assert_eq!(metrics.recall, 0.5);
        assert_eq!(metrics.f1_score, 0.5);
    }

    #[test]
    fn zero_denominators_have_safe_metrics() {
        let metrics = evaluate_binary_classification(&[0.0, 0.0], &[0.0, 0.0]).unwrap();
        assert_eq!(metrics.accuracy, 1.0);
        assert_eq!(metrics.precision, 0.0);
        assert_eq!(metrics.recall, 0.0);
        assert_eq!(metrics.f1_score, 0.0);
    }

    #[test]
    fn rejects_invalid_label_and_empty_input() {
        assert_eq!(
            ConfusionMatrix::from_labels(&[], &[]),
            Err(EvaluationError::EmptyInput)
        );
        assert_eq!(
            ConfusionMatrix::from_labels(&[0.5], &[0.0]),
            Err(EvaluationError::InvalidLabel { value: 0.5 })
        );
    }
}
