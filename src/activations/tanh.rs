use crate::linalg::Vector;

/// Applies the hyperbolic tangent activation to a single value.
pub fn tanh(value: f64) -> f64 {
    value.tanh()
}

/// Applies tanh to every value in a vector.
pub fn tanh_vector(input: &Vector) -> Vector {
    Vector::new(input.iter().map(|value| tanh(*value)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-10, "{left} != {right}");
    }

    #[test]
    fn tanh_of_zero_is_zero() {
        assert_eq!(tanh(0.0), 0.0);
    }

    #[test]
    fn tanh_is_symmetric() {
        assert_close(tanh(1.5), -tanh(-1.5));
    }

    #[test]
    fn tanh_vector_applies_tanh_to_each_value() {
        let input = Vector::new(vec![0.0, 1.0, -1.0]);

        let result = tanh_vector(&input);

        assert_close(result[0], 0.0);
        assert_close(result[1], tanh(1.0));
        assert_close(result[2], tanh(-1.0));
    }
}
