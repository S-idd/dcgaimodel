use crate::linalg::{LinalgError, Vector};
use crate::nn::Neuron;

/// Represents a dense neural-network layer.
///
/// A dense layer contains neurons that all receive the same input vector. Each
/// neuron produces one output value, so the layer output size equals the number
/// of neurons in the layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Creates a dense layer from a list of neurons.
    ///
    /// Returns an error if the neurons do not all expect the same input size.
    pub fn new(neurons: Vec<Neuron>) -> Result<Self, LinalgError> {
        if let Some(first) = neurons.first() {
            let input_size = first.input_size();

            for neuron in &neurons {
                if neuron.input_size() != input_size {
                    return Err(LinalgError::DimensionMismatch {
                        left: input_size,
                        right: neuron.input_size(),
                    });
                }
            }
        }

        Ok(Self { neurons })
    }

    /// Creates a dense layer from weight vectors and bias values.
    ///
    /// Each weight vector and bias pair becomes one neuron.
    pub fn dense(weights: Vec<Vector>, biases: Vec<f64>) -> Result<Self, LinalgError> {
        if weights.len() != biases.len() {
            return Err(LinalgError::DimensionMismatch {
                left: weights.len(),
                right: biases.len(),
            });
        }

        let neurons = weights
            .into_iter()
            .zip(biases)
            .map(|(weights, bias)| Neuron::new(weights, bias))
            .collect();

        Self::new(neurons)
    }

    /// Returns the neurons in this layer.
    pub fn neurons(&self) -> &[Neuron] {
        &self.neurons
    }

    /// Returns mutable access to the neurons in this layer.
    pub fn neurons_mut(&mut self) -> &mut [Neuron] {
        &mut self.neurons
    }

    /// Returns the number of neurons in this layer.
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Returns true when this layer has no neurons.
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Returns the input size expected by this layer.
    pub fn input_size(&self) -> Option<usize> {
        self.neurons.first().map(Neuron::input_size)
    }

    /// Returns the number of output values produced by this layer.
    pub fn output_size(&self) -> usize {
        self.len()
    }

    /// Computes the forward pass for one input vector.
    ///
    /// The output vector contains one value per neuron.
    pub fn forward(&self, inputs: &Vector) -> Result<Vector, LinalgError> {
        let values = self
            .neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect::<Result<Vec<f64>, LinalgError>>()?;

        Ok(Vector::new(values))
    }

    /// Computes the forward pass for a batch of input vectors.
    pub fn forward_batch(&self, batch: &[Vector]) -> Result<Vec<Vector>, LinalgError> {
        batch
            .iter()
            .map(|inputs| self.forward(inputs))
            .collect::<Result<Vec<Vector>, LinalgError>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_dense_layer_from_neurons() {
        let layer = Layer::new(vec![
            Neuron::new(Vector::new(vec![1.0, 2.0]), 0.5),
            Neuron::new(Vector::new(vec![3.0, 4.0]), -0.5),
        ])
        .unwrap();

        assert_eq!(layer.len(), 2);
        assert!(!layer.is_empty());
        assert_eq!(layer.input_size(), Some(2));
        assert_eq!(layer.output_size(), 2);
    }

    #[test]
    fn creates_dense_layer_from_weights_and_biases() {
        let layer = Layer::dense(
            vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])],
            vec![0.5, -0.5],
        )
        .unwrap();

        assert_eq!(layer.neurons().len(), 2);
        assert_eq!(layer.neurons()[0].bias(), 0.5);
        assert_eq!(layer.neurons()[1].bias(), -0.5);
    }

    #[test]
    fn creates_empty_layer() {
        let layer = Layer::new(vec![]).unwrap();

        assert_eq!(layer.len(), 0);
        assert!(layer.is_empty());
        assert_eq!(layer.input_size(), None);
        assert_eq!(layer.output_size(), 0);
    }

    #[test]
    fn rejects_neurons_with_different_input_sizes() {
        let layer = Layer::new(vec![
            Neuron::new(Vector::new(vec![1.0, 2.0]), 0.0),
            Neuron::new(Vector::new(vec![1.0, 2.0, 3.0]), 0.0),
        ]);

        assert_eq!(
            layer,
            Err(LinalgError::DimensionMismatch { left: 2, right: 3 })
        );
    }

    #[test]
    fn rejects_weight_and_bias_count_mismatch() {
        let layer = Layer::dense(vec![Vector::new(vec![1.0, 2.0])], vec![0.0, 1.0]);

        assert_eq!(
            layer,
            Err(LinalgError::DimensionMismatch { left: 1, right: 2 })
        );
    }

    #[test]
    fn forward_computes_one_output_per_neuron() {
        let layer = Layer::dense(
            vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![-1.0, 0.5])],
            vec![0.5, 1.0],
        )
        .unwrap();

        let result = layer.forward(&Vector::new(vec![3.0, 4.0])).unwrap();

        assert_eq!(result, Vector::new(vec![11.5, 0.0]));
    }

    #[test]
    fn forward_dimension_mismatch_returns_error() {
        let layer = Layer::dense(vec![Vector::new(vec![1.0, 2.0])], vec![0.0]).unwrap();

        let result = layer.forward(&Vector::new(vec![1.0]));

        assert_eq!(
            result,
            Err(LinalgError::DimensionMismatch { left: 2, right: 1 })
        );
    }

    #[test]
    fn forward_empty_layer_returns_empty_vector() {
        let layer = Layer::new(vec![]).unwrap();

        let result = layer.forward(&Vector::new(vec![1.0, 2.0])).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn forward_batch_computes_outputs_for_each_input() {
        let layer = Layer::dense(
            vec![Vector::new(vec![1.0, 1.0]), Vector::new(vec![2.0, -1.0])],
            vec![0.0, 0.5],
        )
        .unwrap();

        let batch = vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])];

        let result = layer.forward_batch(&batch).unwrap();

        assert_eq!(
            result,
            vec![Vector::new(vec![3.0, 0.5]), Vector::new(vec![7.0, 2.5])]
        );
    }

    #[test]
    fn forward_batch_dimension_mismatch_returns_error() {
        let layer = Layer::dense(vec![Vector::new(vec![1.0, 2.0])], vec![0.0]).unwrap();
        let batch = vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![1.0])];

        let result = layer.forward_batch(&batch);

        assert_eq!(
            result,
            Err(LinalgError::DimensionMismatch { left: 2, right: 1 })
        );
    }
}
