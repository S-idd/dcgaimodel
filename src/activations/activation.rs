use crate::linalg::Vector;

use super::relu::relu_vector;
use super::sigmoid::sigmoid_vector;
use super::softmax::softmax;
use super::tanh::tanh_vector;

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
        assert_eq!(
            Activation::Sigmoid.derivative_from_output(0.25),
            Some(0.1875)
        );
        assert_eq!(Activation::Tanh.derivative_from_output(0.5), Some(0.75));
        assert_eq!(Activation::Softmax.derivative_from_output(0.5), None);
    }
}
