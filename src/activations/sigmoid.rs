use crate::linalg::Vector;

/// Applies the sigmoid activation to a single value.
///
/// This implementation uses two branches to avoid unnecessary overflow for
/// large positive or negative inputs.
pub fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

/// Applies sigmoid to every value in a vector.
pub fn sigmoid_vector(input: &Vector) -> Vector {
    Vector::new(input.iter().map(|value| sigmoid(*value)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-10, "{left} != {right}");
    }

    #[test]
    fn sigmoid_of_zero_is_half() {
        assert_eq!(sigmoid(0.0), 0.5);
    }

    #[test]
    fn sigmoid_handles_large_values() {
        assert_close(sigmoid(1000.0), 1.0);
        assert_close(sigmoid(-1000.0), 0.0);
    }

    #[test]
    fn sigmoid_vector_applies_sigmoid_to_each_value() {
        let input = Vector::new(vec![0.0, 2.0, -2.0]);

        let result = sigmoid_vector(&input);

        assert_close(result[0], 0.5);
        assert_close(result[1], sigmoid(2.0));
        assert_close(result[2], sigmoid(-2.0));
    }
}
