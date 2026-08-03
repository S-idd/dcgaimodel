use crate::linalg::Vector;

/// Applies the Rectified Linear Unit activation to a single value.
pub fn relu(value: f64) -> f64 {
    value.max(0.0)
}

/// Applies ReLU to every value in a vector.
pub fn relu_vector(input: &Vector) -> Vector {
    Vector::new(input.iter().map(|value| relu(*value)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_returns_positive_values_unchanged() {
        assert_eq!(relu(3.5), 3.5);
    }

    #[test]
    fn relu_clamps_negative_values_to_zero() {
        assert_eq!(relu(-2.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn relu_vector_applies_relu_to_each_value() {
        let input = Vector::new(vec![-2.0, 0.0, 3.0]);

        let result = relu_vector(&input);

        assert_eq!(result, Vector::new(vec![0.0, 0.0, 3.0]));
    }
}
