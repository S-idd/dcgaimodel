use crate::linalg::Vector;

/// Supported activation functions for neural-network layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Returns the input unchanged.
    Linear,

    /// Rectified Linear Unit.
    Relu,

    /// Sigmoid activation.
    Sigmoid,

    /// Hyperbolic tangent activation.
    Tanh,

    /// Softmax activation.
    Softmax,
}

impl Activation {
    /// Applies this activation function to a vector.
    pub fn apply(&self, input: &Vector) -> Vector {
        match self {
            Activation::Linear => input.clone(),
            Activation::Relu => relu_vector(input),
            Activation::Sigmoid => sigmoid_vector(input),
            Activation::Tanh => tanh_vector(input),
            Activation::Softmax => softmax(input),
        }
    }

    /// Returns the activation derivative for one activated output value.
    ///
    /// Softmax needs a full Jacobian for general backpropagation, so it is not
    /// represented as a scalar derivative here.
    pub fn derivative_from_output(&self, output: f64) -> Option<f64> {
        match self {
            Activation::Linear => Some(1.0),
            Activation::Relu => Some(if output > 0.0 { 1.0 } else { 0.0 }),
            Activation::Sigmoid => Some(output * (1.0 - output)),
            Activation::Tanh => Some(1.0 - output * output),
            Activation::Softmax => None,
        }
    }
}

/// Applies the Rectified Linear Unit activation to a single value.
pub fn relu(value: f64) -> f64 {
    value.max(0.0)
}

/// Applies ReLU to every value in a vector.
pub fn relu_vector(input: &Vector) -> Vector {
    Vector::new(input.iter().map(|value| relu(*value)).collect())
}

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

/// Applies the hyperbolic tangent activation to a single value.
pub fn tanh(value: f64) -> f64 {
    value.tanh()
}

/// Applies tanh to every value in a vector.
pub fn tanh_vector(input: &Vector) -> Vector {
    Vector::new(input.iter().map(|value| tanh(*value)).collect())
}

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
    fn linear_activation_returns_input_unchanged() {
        let input = Vector::new(vec![-1.0, 0.0, 1.0]);

        let result = Activation::Linear.apply(&input);

        assert_eq!(result, input);
    }

    #[test]
    fn activation_enum_applies_relu() {
        let input = Vector::new(vec![-2.0, 0.0, 3.0]);

        let result = Activation::Relu.apply(&input);

        assert_eq!(result, Vector::new(vec![0.0, 0.0, 3.0]));
    }

    #[test]
    fn activation_enum_applies_softmax() {
        let input = Vector::new(vec![1.0, 2.0, 3.0]);

        let result = Activation::Softmax.apply(&input);
        let sum = result.iter().sum::<f64>();

        assert_close(sum, 1.0);
    }

    #[test]
    fn activation_derivatives_are_computed_from_outputs() {
        assert_eq!(Activation::Linear.derivative_from_output(3.0), Some(1.0));
        assert_eq!(Activation::Relu.derivative_from_output(2.0), Some(1.0));
        assert_eq!(Activation::Relu.derivative_from_output(0.0), Some(0.0));
        assert_eq!(Activation::Sigmoid.derivative_from_output(0.25), Some(0.1875));
        assert_eq!(Activation::Tanh.derivative_from_output(0.5), Some(0.75));
        assert_eq!(Activation::Softmax.derivative_from_output(0.5), None);
    }

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
