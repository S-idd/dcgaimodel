use crate::linalg::{LinalgError, Vector};

/// Represents a single dense neural-network neuron.
///
/// A neuron stores one weight per input plus a bias term. Its forward pass
/// computes the linear value `weights dot inputs + bias`.
#[derive(Debug, Clone, PartialEq)]
pub struct Neuron {
    weights: Vector,
    bias: f64,
}

impl Neuron {
    /// Creates a new neuron with the provided weights and bias.
    pub fn new(weights: Vector, bias: f64) -> Self {
        Self { weights, bias }
    }

    /// Returns the neuron's weights.
    pub fn weights(&self) -> &Vector {
        &self.weights
    }

    /// Returns the neuron's bias.
    pub fn bias(&self) -> f64 {
        self.bias
    }

    /// Returns the number of inputs expected by this neuron.
    pub fn input_size(&self) -> usize {
        self.weights.len()
    }

    /// Computes the neuron's linear forward pass.
    ///
    /// Returns an error if the input vector length does not match the weight
    /// vector length.
    pub fn forward(&self, inputs: &Vector) -> Result<f64, LinalgError> {
        Ok(self.weights.dot_product(inputs)? + self.bias)
    }

    /// Applies a gradient update to this neuron's weights and bias.
    pub fn apply_gradients(
        &mut self,
        weight_gradients: &Vector,
        bias_gradient: f64,
        learning_rate: f64,
    ) -> Result<(), LinalgError> {
        if self.weights.len() != weight_gradients.len() {
            return Err(LinalgError::DimensionMismatch {
                left: self.weights.len(),
                right: weight_gradients.len(),
            });
        }

        let updated_weights = self
            .weights
            .iter()
            .zip(weight_gradients.iter())
            .map(|(weight, gradient)| weight - learning_rate * gradient)
            .collect();

        self.weights = Vector::new(updated_weights);
        self.bias -= learning_rate * bias_gradient;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_neuron_successfully() {
        let weights = Vector::new(vec![0.5, -1.0, 2.0]);
        let neuron = Neuron::new(weights.clone(), 0.25);

        assert_eq!(neuron.weights(), &weights);
        assert_eq!(neuron.bias(), 0.25);
        assert_eq!(neuron.input_size(), 3);
    }

    #[test]
    fn forward_computes_weighted_sum_plus_bias() {
        let neuron = Neuron::new(Vector::new(vec![0.5, -1.0, 2.0]), 0.25);
        let inputs = Vector::new(vec![4.0, 3.0, 2.0]);

        let result = neuron.forward(&inputs).unwrap();

        assert_eq!(result, 3.25);
    }

    #[test]
    fn forward_dimension_mismatch_returns_error() {
        let neuron = Neuron::new(Vector::new(vec![1.0, 2.0]), 0.0);
        let inputs = Vector::new(vec![1.0]);

        let result = neuron.forward(&inputs);

        assert_eq!(
            result,
            Err(LinalgError::DimensionMismatch { left: 2, right: 1 })
        );
    }

    #[test]
    fn applies_weight_and_bias_gradients() {
        let mut neuron = Neuron::new(Vector::new(vec![1.0, -2.0]), 0.5);

        neuron
            .apply_gradients(&Vector::new(vec![0.25, -0.5]), 1.0, 0.1)
            .unwrap();

        assert_eq!(neuron.weights(), &Vector::new(vec![0.975, -1.95]));
        assert_eq!(neuron.bias(), 0.4);
    }

    #[test]
    fn apply_gradients_dimension_mismatch_returns_error() {
        let mut neuron = Neuron::new(Vector::new(vec![1.0, 2.0]), 0.0);

        let result = neuron.apply_gradients(&Vector::new(vec![1.0]), 0.0, 0.1);

        assert_eq!(
            result,
            Err(LinalgError::DimensionMismatch { left: 2, right: 1 })
        );
    }
}
