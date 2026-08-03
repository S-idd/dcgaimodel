use crate::activations::Activation;
use crate::linalg::{LinalgError, Vector};
use crate::nn::Layer;
use std::error::Error;
use std::fmt;

/// Errors that can occur while building or running a neural network.
#[derive(Debug, PartialEq)]
pub enum NetworkError {
    /// A network must contain at least one layer.
    EmptyNetwork,

    /// A network layer cannot be empty.
    EmptyLayer { index: usize },

    /// Adjacent layers or inputs have incompatible dimensions.
    DimensionMismatch {
        layer_index: usize,
        expected: usize,
        actual: usize,
    },

    /// A lower-level linear algebra operation failed.
    Linalg(LinalgError),
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetworkError::EmptyNetwork => write!(f, "Network must contain at least one layer."),
            NetworkError::EmptyLayer { index } => {
                write!(f, "Network layer {} cannot be empty.", index)
            }
            NetworkError::DimensionMismatch {
                layer_index,
                expected,
                actual,
            } => write!(
                f,
                "Dimension mismatch at layer {}: expected {}, got {}",
                layer_index, expected, actual
            ),
            NetworkError::Linalg(error) => write!(f, "Linear algebra error: {}", error),
        }
    }
}

impl Error for NetworkError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            NetworkError::Linalg(error) => Some(error),
            _ => None,
        }
    }
}

impl From<LinalgError> for NetworkError {
    fn from(error: LinalgError) -> Self {
        NetworkError::Linalg(error)
    }
}

/// A dense layer paired with the activation that should run after it.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkLayer {
    layer: Layer,
    activation: Activation,
}

impl NetworkLayer {
    /// Creates a network layer from a dense layer and activation.
    pub fn new(layer: Layer, activation: Activation) -> Self {
        Self { layer, activation }
    }

    /// Returns the dense layer.
    pub fn layer(&self) -> &Layer {
        &self.layer
    }

    /// Returns the activation function.
    pub fn activation(&self) -> Activation {
        self.activation
    }

    /// Returns the input size expected by this layer.
    pub fn input_size(&self) -> Option<usize> {
        self.layer.input_size()
    }

    /// Returns the output size produced by this layer.
    pub fn output_size(&self) -> usize {
        self.layer.output_size()
    }

    /// Runs the dense layer followed by its activation.
    pub fn forward(&self, inputs: &Vector) -> Result<Vector, NetworkError> {
        let values = self.layer.forward(inputs)?;

        Ok(self.activation.apply(&values))
    }
}

/// A sequential feed-forward neural network.
#[derive(Debug, Clone, PartialEq)]
pub struct Network {
    layers: Vec<NetworkLayer>,
}

impl Network {
    /// Creates a sequential network and validates adjacent layer dimensions.
    pub fn new(layers: Vec<NetworkLayer>) -> Result<Self, NetworkError> {
        validate_layers(&layers)?;

        Ok(Self { layers })
    }

    /// Returns the network layers.
    pub fn layers(&self) -> &[NetworkLayer] {
        &self.layers
    }

    /// Returns the number of layers in the network.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true when the network has no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Returns the input size expected by the first layer.
    pub fn input_size(&self) -> usize {
        self.layers
            .first()
            .and_then(NetworkLayer::input_size)
            .unwrap_or(0)
    }

    /// Returns the output size produced by the final layer.
    pub fn output_size(&self) -> usize {
        self.layers
            .last()
            .map(NetworkLayer::output_size)
            .unwrap_or(0)
    }

    /// Runs forward propagation for one input vector.
    pub fn forward(&self, inputs: &Vector) -> Result<Vector, NetworkError> {
        let expected = self.input_size();

        if inputs.len() != expected {
            return Err(NetworkError::DimensionMismatch {
                layer_index: 0,
                expected,
                actual: inputs.len(),
            });
        }

        let mut output = inputs.clone();

        for layer in &self.layers {
            output = layer.forward(&output)?;
        }

        Ok(output)
    }

    /// Runs forward propagation for a batch of input vectors.
    pub fn forward_batch(&self, batch: &[Vector]) -> Result<Vec<Vector>, NetworkError> {
        batch
            .iter()
            .map(|inputs| self.forward(inputs))
            .collect::<Result<Vec<Vector>, NetworkError>>()
    }
}

fn validate_layers(layers: &[NetworkLayer]) -> Result<(), NetworkError> {
    if layers.is_empty() {
        return Err(NetworkError::EmptyNetwork);
    }

    for (index, layer) in layers.iter().enumerate() {
        if layer.layer().is_empty() {
            return Err(NetworkError::EmptyLayer { index });
        }
    }

    for (index, pair) in layers.windows(2).enumerate() {
        let previous_output = pair[0].output_size();
        let next_input = pair[1].input_size().unwrap_or(0);

        if previous_output != next_input {
            return Err(NetworkError::DimensionMismatch {
                layer_index: index + 1,
                expected: previous_output,
                actual: next_input,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Vector;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-10, "{left} != {right}");
    }

    fn dense_layer(weights: Vec<Vec<f64>>, biases: Vec<f64>) -> Layer {
        let weights = weights.into_iter().map(Vector::new).collect();

        Layer::dense(weights, biases).unwrap()
    }

    #[test]
    fn creates_network_successfully() {
        let network = Network::new(vec![
            NetworkLayer::new(
                dense_layer(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0.0, 0.0]),
                Activation::Relu,
            ),
            NetworkLayer::new(
                dense_layer(vec![vec![1.0, 1.0]], vec![0.0]),
                Activation::Linear,
            ),
        ])
        .unwrap();

        assert_eq!(network.len(), 2);
        assert!(!network.is_empty());
        assert_eq!(network.input_size(), 2);
        assert_eq!(network.output_size(), 1);
        assert_eq!(network.layers()[0].activation(), Activation::Relu);
    }

    #[test]
    fn rejects_empty_network() {
        let result = Network::new(vec![]);

        assert_eq!(result, Err(NetworkError::EmptyNetwork));
    }

    #[test]
    fn rejects_empty_layer() {
        let result = Network::new(vec![NetworkLayer::new(
            Layer::new(vec![]).unwrap(),
            Activation::Linear,
        )]);

        assert_eq!(result, Err(NetworkError::EmptyLayer { index: 0 }));
    }

    #[test]
    fn rejects_incompatible_adjacent_layers() {
        let result = Network::new(vec![
            NetworkLayer::new(
                dense_layer(vec![vec![1.0, 1.0]], vec![0.0]),
                Activation::Relu,
            ),
            NetworkLayer::new(
                dense_layer(vec![vec![1.0, 1.0]], vec![0.0]),
                Activation::Linear,
            ),
        ]);

        assert_eq!(
            result,
            Err(NetworkError::DimensionMismatch {
                layer_index: 1,
                expected: 1,
                actual: 2
            })
        );
    }

    #[test]
    fn network_layer_runs_dense_forward_then_activation() {
        let layer = NetworkLayer::new(
            dense_layer(vec![vec![1.0, -2.0], vec![2.0, 1.0]], vec![0.0, -1.0]),
            Activation::Relu,
        );

        let result = layer.forward(&Vector::new(vec![1.0, 2.0])).unwrap();

        assert_eq!(result, Vector::new(vec![0.0, 3.0]));
    }

    #[test]
    fn forward_runs_through_all_layers() {
        let network = Network::new(vec![
            NetworkLayer::new(
                dense_layer(vec![vec![1.0, -1.0], vec![0.5, 0.5]], vec![0.0, 0.0]),
                Activation::Relu,
            ),
            NetworkLayer::new(
                dense_layer(vec![vec![1.0, 2.0]], vec![0.5]),
                Activation::Linear,
            ),
        ])
        .unwrap();

        let result = network.forward(&Vector::new(vec![4.0, 2.0])).unwrap();

        assert_eq!(result, Vector::new(vec![8.5]));
    }

    #[test]
    fn forward_supports_softmax_output_layer() {
        let network = Network::new(vec![NetworkLayer::new(
            dense_layer(
                vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
                vec![0.0, 0.0, 0.0],
            ),
            Activation::Softmax,
        )])
        .unwrap();

        let result = network.forward(&Vector::new(vec![1.0, 2.0])).unwrap();
        let sum = result.iter().sum::<f64>();

        assert_close(sum, 1.0);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn forward_input_dimension_mismatch_returns_error() {
        let network = Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![1.0, 1.0]], vec![0.0]),
            Activation::Linear,
        )])
        .unwrap();

        let result = network.forward(&Vector::new(vec![1.0]));

        assert_eq!(
            result,
            Err(NetworkError::DimensionMismatch {
                layer_index: 0,
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn forward_batch_runs_each_input_through_network() {
        let network = Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![1.0, 1.0], vec![2.0, -1.0]], vec![0.0, 0.5]),
            Activation::Linear,
        )])
        .unwrap();

        let batch = vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])];

        let result = network.forward_batch(&batch).unwrap();

        assert_eq!(
            result,
            vec![Vector::new(vec![3.0, 0.5]), Vector::new(vec![7.0, 2.5])]
        );
    }

    #[test]
    fn forward_batch_dimension_mismatch_returns_error() {
        let network = Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![1.0, 1.0]], vec![0.0]),
            Activation::Linear,
        )])
        .unwrap();
        let batch = vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![1.0])];

        let result = network.forward_batch(&batch);

        assert_eq!(
            result,
            Err(NetworkError::DimensionMismatch {
                layer_index: 0,
                expected: 2,
                actual: 1
            })
        );
    }
}
