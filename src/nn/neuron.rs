use crate::linalg::{LinalgError, Vector};

/// Cached intermediate values produced while running a neuron's forward pass
/// during training.
///
/// Backpropagation needs the input that was fed into the neuron, the
/// pre-activation value `z = weights · input + bias`, and the post-activation
/// output. Caching them here avoids recomputing them during the backward
/// pass. The neuron itself has no notion of activation functions — the
/// owning layer computes the activation and reports it back via
/// [`Neuron::cache_activation`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NeuronCache {
    input: Option<Vector>,
    pre_activation: Option<f64>,
    activation: Option<f64>,
}

impl NeuronCache {
    /// Returns the input vector from the most recent cached forward pass.
    pub fn input(&self) -> Option<&Vector> {
        self.input.as_ref()
    }

    /// Returns the cached pre-activation value `z`.
    pub fn pre_activation(&self) -> Option<f64> {
        self.pre_activation
    }

    /// Returns the cached post-activation output, if one has been set.
    pub fn activation(&self) -> Option<f64> {
        self.activation
    }

    fn clear(&mut self) {
        self.input = None;
        self.pre_activation = None;
        self.activation = None;
    }
}

/// Represents a single dense neural-network neuron.
///
/// A neuron stores one weight per input plus a bias term. Its forward pass
/// computes the linear value `weights dot inputs + bias`. During training,
/// the neuron can also cache the values needed for backpropagation (see
/// [`Neuron::forward_with_cache`] and [`Neuron::cache_activation`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Neuron {
    weights: Vector,
    bias: f64,
    cache: NeuronCache,
}

impl Neuron {
    /// Creates a new neuron with the provided weights and bias.
    pub fn new(weights: Vector, bias: f64) -> Self {
        Self {
            weights,
            bias,
            cache: NeuronCache::default(),
        }
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

    /// Returns the neuron's cached training values, if any.
    pub fn cache(&self) -> &NeuronCache {
        &self.cache
    }

    /// Clears any cached training values.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Computes the neuron's linear forward pass.
    ///
    /// Returns an error if the input vector length does not match the weight
    /// vector length.
    pub fn forward(&self, inputs: &Vector) -> Result<f64, LinalgError> {
        Ok(self.weights.dot_product(inputs)? + self.bias)
    }

    /// Computes the forward pass and caches the input and pre-activation
    /// value (`z`) for later use during backpropagation.
    ///
    /// Any previously cached activation output is cleared, since it no
    /// longer corresponds to this input. Call [`Neuron::cache_activation`]
    /// once the owning layer has applied its activation function to `z`.
    pub fn forward_with_cache(&mut self, inputs: &Vector) -> Result<f64, LinalgError> {
        let pre_activation = self.forward(inputs)?;

        self.cache.input = Some(inputs.clone());
        self.cache.pre_activation = Some(pre_activation);
        self.cache.activation = None;

        Ok(pre_activation)
    }

    /// Stores the post-activation output for the most recently cached
    /// forward pass. Intended to be called by the owning layer after it
    /// applies an activation function to this neuron's pre-activation value.
    pub fn cache_activation(&mut self, activation_output: f64) {
        self.cache.activation = Some(activation_output);
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
        assert_eq!(neuron.cache(), &NeuronCache::default());
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
    fn forward_with_cache_stores_input_and_pre_activation() {
        let mut neuron = Neuron::new(Vector::new(vec![0.5, -1.0, 2.0]), 0.25);
        let inputs = Vector::new(vec![4.0, 3.0, 2.0]);

        let result = neuron.forward_with_cache(&inputs).unwrap();

        assert_eq!(result, 3.25);
        assert_eq!(neuron.cache().input(), Some(&inputs));
        assert_eq!(neuron.cache().pre_activation(), Some(3.25));
        assert_eq!(neuron.cache().activation(), None);
    }

    #[test]
    fn cache_activation_stores_post_activation_output() {
        let mut neuron = Neuron::new(Vector::new(vec![1.0]), 0.0);
        let inputs = Vector::new(vec![1.0]);

        neuron.forward_with_cache(&inputs).unwrap();
        neuron.cache_activation(0.7310585786300049);

        assert_eq!(neuron.cache().activation(), Some(0.7310585786300049));
    }

    #[test]
    fn forward_with_cache_clears_stale_activation_on_new_input() {
        let mut neuron = Neuron::new(Vector::new(vec![1.0]), 0.0);

        neuron.forward_with_cache(&Vector::new(vec![1.0])).unwrap();
        neuron.cache_activation(0.5);

        neuron.forward_with_cache(&Vector::new(vec![2.0])).unwrap();

        assert_eq!(neuron.cache().activation(), None);
        assert_eq!(neuron.cache().pre_activation(), Some(2.0));
    }

    #[test]
    fn clear_cache_resets_all_cached_values() {
        let mut neuron = Neuron::new(Vector::new(vec![1.0]), 0.0);
        neuron.forward_with_cache(&Vector::new(vec![1.0])).unwrap();
        neuron.cache_activation(0.5);

        neuron.clear_cache();

        assert_eq!(neuron.cache(), &NeuronCache::default());
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
