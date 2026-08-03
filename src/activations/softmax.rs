use crate::linalg::Vector;

/// Applies softmax to a vector of logits.
///
/// The logits are shifted by their maximum value before exponentiation. This
/// keeps the output stable for large values without changing the probabilities.
pub fn softmax(input: &Vector) -> Vector {
    if input.is_empty() {
        return Vector::new(vec![]);
    }

    let max = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_values = input
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<f64>>();
    let sum = exp_values.iter().sum::<f64>();
    let probabilities = exp_values.iter().map(|value| value / sum).collect();

    Vector::new(probabilities)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-10, "{left} != {right}");
    }

    #[test]
    fn softmax_returns_empty_vector_for_empty_input() {
        let result = softmax(&Vector::new(vec![]));

        assert!(result.is_empty());
    }

    #[test]
    fn softmax_outputs_probabilities_that_sum_to_one() {
        let input = Vector::new(vec![1.0, 2.0, 3.0]);

        let result = softmax(&input);
        let sum = result.iter().sum::<f64>();

        assert_close(sum, 1.0);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let input = Vector::new(vec![1000.0, 1001.0, 1002.0]);

        let result = softmax(&input);
        let sum = result.iter().sum::<f64>();

        assert_close(sum, 1.0);
        assert!(result.iter().all(|value| value.is_finite()));
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }
}
