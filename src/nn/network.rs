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

    /// A layer activation does not support this backpropagation path yet.
    UnsupportedActivation {
        layer_index: usize,
        activation: Activation,
    },

    /// Softmax cross-entropy requires a normalized, non-negative target vector.
    InvalidTargetDistribution { sum: f64 },

    /// A softmax cross-entropy target contained an invalid probability.
    InvalidTarget { value: f64 },

    /// Learning rate must be positive and finite.
    InvalidLearningRate { value: f64 },

    /// Two layer gradients contain different numbers of neuron gradients.
    GradientNeuronCountMismatch { left: usize, right: usize },

    /// Two network gradients contain different numbers of layer gradients.
    GradientLayerCountMismatch { left: usize, right: usize },

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
            NetworkError::UnsupportedActivation {
                layer_index,
                activation,
            } => write!(
                f,
                "Activation {:?} at layer {} is not supported for this backpropagation path.",
                activation, layer_index
            ),
            NetworkError::InvalidTargetDistribution { sum } => {
                write!(f, "Target probabilities must sum to 1, got {sum}")
            }
            NetworkError::InvalidTarget { value } => {
                write!(
                    f,
                    "Target probability must be finite and in 0..=1, got {value}"
                )
            }
            NetworkError::InvalidLearningRate { value } => {
                write!(f, "Learning rate must be positive and finite: {}", value)
            }
            NetworkError::GradientNeuronCountMismatch { left, right } => write!(
                f,
                "Gradient neuron-count mismatch: left = {}, right = {}",
                left, right
            ),
            NetworkError::GradientLayerCountMismatch { left, right } => write!(
                f,
                "Gradient layer-count mismatch: left = {}, right = {}",
                left, right
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

/// Gradients for one dense layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerGradient {
    weight_gradients: Vec<Vector>,
    bias_gradients: Vector,
}

impl LayerGradient {
    /// Creates gradients for one dense layer.
    pub fn new(weight_gradients: Vec<Vector>, bias_gradients: Vector) -> Self {
        Self {
            weight_gradients,
            bias_gradients,
        }
    }

    /// Returns one weight-gradient vector per neuron.
    pub fn weight_gradients(&self) -> &[Vector] {
        &self.weight_gradients
    }

    /// Returns one bias gradient per neuron.
    pub fn bias_gradients(&self) -> &Vector {
        &self.bias_gradients
    }

    /// Adds another layer gradient element-wise.
    ///
    /// Returns an error when the gradients contain different numbers of
    /// neuron gradients or when corresponding vectors have different lengths.
    pub fn add(&self, other: &Self) -> Result<Self, NetworkError> {
        if self.weight_gradients.len() != other.weight_gradients.len() {
            return Err(NetworkError::GradientNeuronCountMismatch {
                left: self.weight_gradients.len(),
                right: other.weight_gradients.len(),
            });
        }

        let weight_gradients = self
            .weight_gradients
            .iter()
            .zip(other.weight_gradients.iter())
            .map(|(left, right)| left.add(right))
            .collect::<Result<Vec<_>, _>>()?;
        let bias_gradients = self.bias_gradients.add(&other.bias_gradients)?;

        Ok(Self::new(weight_gradients, bias_gradients))
    }

    /// Returns a copy with every gradient value multiplied by `factor`.
    pub fn scale(&self, factor: f64) -> Self {
        Self::new(
            self.weight_gradients
                .iter()
                .map(|gradient| gradient.scalar_multiply(factor))
                .collect(),
            self.bias_gradients.scalar_multiply(factor),
        )
    }
}

/// Gradients for an entire sequential network.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkGradients {
    layer_gradients: Vec<LayerGradient>,
}

impl NetworkGradients {
    /// Creates network gradients.
    pub fn new(layer_gradients: Vec<LayerGradient>) -> Self {
        Self { layer_gradients }
    }

    /// Returns one gradient group per network layer.
    pub fn layer_gradients(&self) -> &[LayerGradient] {
        &self.layer_gradients
    }

    /// Returns the number of gradient groups.
    pub fn len(&self) -> usize {
        self.layer_gradients.len()
    }

    /// Returns true when no gradients are present.
    pub fn is_empty(&self) -> bool {
        self.layer_gradients.is_empty()
    }

    /// Adds another network gradient element-wise.
    ///
    /// Returns an error when the gradients contain different numbers of
    /// layers or when a corresponding layer gradient has an incompatible
    /// shape.
    pub fn add(&self, other: &Self) -> Result<Self, NetworkError> {
        if self.len() != other.len() {
            return Err(NetworkError::GradientLayerCountMismatch {
                left: self.len(),
                right: other.len(),
            });
        }

        let layer_gradients = self
            .layer_gradients
            .iter()
            .zip(other.layer_gradients.iter())
            .map(|(left, right)| left.add(right))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self::new(layer_gradients))
    }

    /// Returns a copy with every layer gradient multiplied by `factor`.
    pub fn scale(&self, factor: f64) -> Self {
        Self::new(
            self.layer_gradients
                .iter()
                .map(|gradient| gradient.scale(factor))
                .collect(),
        )
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

    /// Returns mutable access to the network layers.
    pub fn layers_mut(&mut self) -> &mut [NetworkLayer] {
        &mut self.layers
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

    /// Computes MSE gradients for one input-target pair.
    pub fn backpropagate_mse(
        &self,
        inputs: &Vector,
        target: &Vector,
    ) -> Result<NetworkGradients, NetworkError> {
        let (prediction, trace) = self.forward_with_trace(inputs)?;

        if prediction.len() != target.len() {
            return Err(NetworkError::DimensionMismatch {
                layer_index: self.len(),
                expected: prediction.len(),
                actual: target.len(),
            });
        }

        let output_scale = 2.0 / prediction.len() as f64;
        let mut upstream_gradients = prediction
            .iter()
            .zip(target.iter())
            .map(|(predicted, expected)| output_scale * (predicted - expected))
            .collect::<Vec<f64>>();

        let mut layer_gradients = Vec::with_capacity(self.len());

        for layer_index in (0..self.layers.len()).rev() {
            let network_layer = &self.layers[layer_index];
            let layer_trace = &trace[layer_index];
            let activated_output = &layer_trace.output;
            let layer_input = &layer_trace.input;

            let deltas = upstream_gradients
                .iter()
                .zip(activated_output.iter())
                .map(|(gradient, output)| {
                    network_layer
                        .activation()
                        .derivative_from_output(*output)
                        .map(|derivative| gradient * derivative)
                        .ok_or(NetworkError::UnsupportedActivation {
                            layer_index,
                            activation: network_layer.activation(),
                        })
                })
                .collect::<Result<Vec<f64>, NetworkError>>()?;

            let weight_gradients = deltas
                .iter()
                .map(|delta| {
                    Vector::new(
                        layer_input
                            .iter()
                            .map(|input_value| delta * input_value)
                            .collect(),
                    )
                })
                .collect::<Vec<Vector>>();

            let bias_gradients = Vector::new(deltas.clone());

            upstream_gradients = vec![0.0; network_layer.input_size().unwrap_or(0)];

            for (neuron, delta) in network_layer.layer().neurons().iter().zip(deltas.iter()) {
                for (input_index, weight) in neuron.weights().iter().enumerate() {
                    upstream_gradients[input_index] += delta * weight;
                }
            }

            layer_gradients.push(LayerGradient::new(weight_gradients, bias_gradients));
        }

        layer_gradients.reverse();

        Ok(NetworkGradients::new(layer_gradients))
    }

    /// Computes gradients for a Softmax output layer with Cross-Entropy loss.
    ///
    /// This uses the mathematically fused output derivative `prediction - target`.
    /// Hidden layers use their ordinary activation derivatives. It intentionally
    /// rejects any network whose final activation is not Softmax.
    pub fn backpropagate_softmax_cross_entropy(
        &self,
        inputs: &Vector,
        target: &Vector,
    ) -> Result<NetworkGradients, NetworkError> {
        let (prediction, trace) = self.forward_with_trace(inputs)?;
        if prediction.len() != target.len() {
            return Err(NetworkError::DimensionMismatch {
                layer_index: self.len(),
                expected: prediction.len(),
                actual: target.len(),
            });
        }
        let output_index = self.layers.len() - 1;
        if self.layers[output_index].activation() != Activation::Softmax {
            return Err(NetworkError::UnsupportedActivation {
                layer_index: output_index,
                activation: self.layers[output_index].activation(),
            });
        }
        let target_sum = target.iter().sum::<f64>();
        if target
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(NetworkError::InvalidTarget {
                value: target
                    .iter()
                    .copied()
                    .find(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                    .unwrap_or(f64::NAN),
            });
        }
        if (target_sum - 1.0).abs() > 1e-9 {
            return Err(NetworkError::InvalidTargetDistribution { sum: target_sum });
        }

        let mut upstream_gradients = prediction
            .iter()
            .zip(target.iter())
            .map(|(predicted, expected)| predicted - expected)
            .collect::<Vec<_>>();
        let mut layer_gradients = Vec::with_capacity(self.len());

        for layer_index in (0..self.layers.len()).rev() {
            let network_layer = &self.layers[layer_index];
            let layer_trace = &trace[layer_index];
            let deltas = if layer_index == output_index {
                upstream_gradients.clone()
            } else {
                upstream_gradients
                    .iter()
                    .zip(layer_trace.output.iter())
                    .map(|(gradient, output)| {
                        network_layer
                            .activation()
                            .derivative_from_output(*output)
                            .map(|derivative| gradient * derivative)
                            .ok_or(NetworkError::UnsupportedActivation {
                                layer_index,
                                activation: network_layer.activation(),
                            })
                    })
                    .collect::<Result<Vec<_>, NetworkError>>()?
            };
            let weight_gradients = deltas
                .iter()
                .map(|delta| {
                    Vector::new(
                        layer_trace
                            .input
                            .iter()
                            .map(|input| delta * input)
                            .collect(),
                    )
                })
                .collect();
            let bias_gradients = Vector::new(deltas.clone());
            upstream_gradients = vec![0.0; network_layer.input_size().unwrap_or(0)];
            for (neuron, delta) in network_layer.layer().neurons().iter().zip(&deltas) {
                for (input_index, weight) in neuron.weights().iter().enumerate() {
                    upstream_gradients[input_index] += delta * weight;
                }
            }
            layer_gradients.push(LayerGradient::new(weight_gradients, bias_gradients));
        }
        layer_gradients.reverse();
        Ok(NetworkGradients::new(layer_gradients))
    }

    /// Applies network gradients using vanilla gradient descent.
    pub fn apply_gradients(
        &mut self,
        gradients: &NetworkGradients,
        learning_rate: f64,
    ) -> Result<(), NetworkError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(NetworkError::InvalidLearningRate {
                value: learning_rate,
            });
        }

        if gradients.len() != self.len() {
            return Err(NetworkError::DimensionMismatch {
                layer_index: 0,
                expected: self.len(),
                actual: gradients.len(),
            });
        }

        for (layer_index, (network_layer, layer_gradient)) in self
            .layers
            .iter_mut()
            .zip(gradients.layer_gradients().iter())
            .enumerate()
        {
            if layer_gradient.weight_gradients().len() != network_layer.output_size() {
                return Err(NetworkError::DimensionMismatch {
                    layer_index,
                    expected: network_layer.output_size(),
                    actual: layer_gradient.weight_gradients().len(),
                });
            }

            if layer_gradient.bias_gradients().len() != network_layer.output_size() {
                return Err(NetworkError::DimensionMismatch {
                    layer_index,
                    expected: network_layer.output_size(),
                    actual: layer_gradient.bias_gradients().len(),
                });
            }

            for (neuron_index, neuron) in network_layer.layer.neurons_mut().iter_mut().enumerate() {
                neuron.apply_gradients(
                    &layer_gradient.weight_gradients()[neuron_index],
                    layer_gradient.bias_gradients()[neuron_index],
                    learning_rate,
                )?;
            }
        }

        Ok(())
    }

    /// Runs one MSE training step and returns the gradients that were applied.
    pub fn train_mse(
        &mut self,
        inputs: &Vector,
        target: &Vector,
        learning_rate: f64,
    ) -> Result<NetworkGradients, NetworkError> {
        let gradients = self.backpropagate_mse(inputs, target)?;

        self.apply_gradients(&gradients, learning_rate)?;

        Ok(gradients)
    }

    fn forward_with_trace(
        &self,
        inputs: &Vector,
    ) -> Result<(Vector, Vec<LayerForwardTrace>), NetworkError> {
        let expected = self.input_size();

        if inputs.len() != expected {
            return Err(NetworkError::DimensionMismatch {
                layer_index: 0,
                expected,
                actual: inputs.len(),
            });
        }

        let mut output = inputs.clone();
        let mut trace = Vec::with_capacity(self.len());

        for layer in &self.layers {
            let layer_input = output;
            output = layer.forward(&layer_input)?;
            trace.push(LayerForwardTrace {
                input: layer_input,
                output: output.clone(),
            });
        }

        Ok((output, trace))
    }
}

#[derive(Debug, Clone, PartialEq)]
struct LayerForwardTrace {
    input: Vector,
    output: Vector,
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

    fn sample_layer_gradient() -> LayerGradient {
        LayerGradient::new(
            vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])],
            Vector::new(vec![5.0, 6.0]),
        )
    }

    #[test]
    fn creates_layer_gradient_successfully() {
        let gradient = sample_layer_gradient();

        assert_eq!(gradient.weight_gradients().len(), 2);
        assert_eq!(gradient.bias_gradients().len(), 2);
    }

    #[test]
    fn layer_gradient_accessors_return_values() {
        let gradient = sample_layer_gradient();

        assert_eq!(
            gradient.weight_gradients(),
            &[Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])]
        );
        assert_eq!(gradient.bias_gradients(), &Vector::new(vec![5.0, 6.0]));
    }

    #[test]
    fn layer_gradient_adds_compatible_gradients() {
        let left = sample_layer_gradient();
        let right = LayerGradient::new(
            vec![Vector::new(vec![10.0, 20.0]), Vector::new(vec![30.0, 40.0])],
            Vector::new(vec![50.0, 60.0]),
        );

        let result = left.add(&right).unwrap();

        assert_eq!(
            result,
            LayerGradient::new(
                vec![Vector::new(vec![11.0, 22.0]), Vector::new(vec![33.0, 44.0])],
                Vector::new(vec![55.0, 66.0]),
            )
        );
    }

    #[test]
    fn layer_gradient_add_rejects_incompatible_neuron_counts() {
        let left = LayerGradient::new(
            vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])],
            Vector::new(vec![1.0, 2.0]),
        );
        let right = LayerGradient::new(vec![Vector::new(vec![3.0])], Vector::new(vec![3.0]));

        assert_eq!(
            left.add(&right),
            Err(NetworkError::GradientNeuronCountMismatch { left: 2, right: 1 })
        );
    }

    #[test]
    fn layer_gradient_add_rejects_incompatible_weight_vector_sizes() {
        let left = LayerGradient::new(vec![Vector::new(vec![1.0, 2.0])], Vector::new(vec![3.0]));
        let right = LayerGradient::new(vec![Vector::new(vec![4.0])], Vector::new(vec![5.0]));

        assert_eq!(
            left.add(&right),
            Err(NetworkError::Linalg(LinalgError::DimensionMismatch {
                left: 2,
                right: 1,
            }))
        );
    }

    #[test]
    fn layer_gradient_add_rejects_incompatible_bias_sizes() {
        let left = LayerGradient::new(vec![Vector::new(vec![1.0])], Vector::new(vec![2.0, 3.0]));
        let right = LayerGradient::new(vec![Vector::new(vec![4.0])], Vector::new(vec![5.0]));

        assert_eq!(
            left.add(&right),
            Err(NetworkError::Linalg(LinalgError::DimensionMismatch {
                left: 2,
                right: 1,
            }))
        );
    }

    #[test]
    fn layer_gradient_scale_by_one_preserves_values() {
        let gradient = sample_layer_gradient();

        assert_eq!(gradient.scale(1.0), gradient);
    }

    #[test]
    fn layer_gradient_scale_by_zero_zeros_every_value() {
        let result = sample_layer_gradient().scale(0.0);

        assert_eq!(
            result,
            LayerGradient::new(
                vec![Vector::new(vec![0.0, 0.0]), Vector::new(vec![0.0, 0.0])],
                Vector::new(vec![0.0, 0.0]),
            )
        );
    }

    #[test]
    fn layer_gradient_scale_by_half_averages_values() {
        let result = sample_layer_gradient().scale(0.5);

        assert_eq!(
            result,
            LayerGradient::new(
                vec![Vector::new(vec![0.5, 1.0]), Vector::new(vec![1.5, 2.0])],
                Vector::new(vec![2.5, 3.0]),
            )
        );
    }

    #[test]
    fn layer_gradient_scale_by_negative_value_negates_values() {
        let result = sample_layer_gradient().scale(-1.0);

        assert_eq!(
            result,
            LayerGradient::new(
                vec![Vector::new(vec![-1.0, -2.0]), Vector::new(vec![-3.0, -4.0])],
                Vector::new(vec![-5.0, -6.0]),
            )
        );
    }

    #[test]
    fn creates_network_gradients_successfully() {
        let gradients = NetworkGradients::new(vec![sample_layer_gradient()]);

        assert_eq!(gradients.len(), 1);
        assert!(!gradients.is_empty());
    }

    #[test]
    fn network_gradient_accessors_return_values() {
        let layer_gradient = sample_layer_gradient();
        let gradients = NetworkGradients::new(vec![layer_gradient.clone()]);

        assert_eq!(gradients.layer_gradients(), &[layer_gradient]);
    }

    #[test]
    fn network_gradient_reports_empty_state() {
        let gradients = NetworkGradients::new(vec![]);

        assert_eq!(gradients.len(), 0);
        assert!(gradients.is_empty());
    }

    #[test]
    fn network_gradients_add_compatible_layers() {
        let left = NetworkGradients::new(vec![sample_layer_gradient()]);
        let right = NetworkGradients::new(vec![LayerGradient::new(
            vec![Vector::new(vec![10.0, 20.0]), Vector::new(vec![30.0, 40.0])],
            Vector::new(vec![50.0, 60.0]),
        )]);

        let result = left.add(&right).unwrap();

        assert_eq!(
            result,
            NetworkGradients::new(vec![LayerGradient::new(
                vec![Vector::new(vec![11.0, 22.0]), Vector::new(vec![33.0, 44.0])],
                Vector::new(vec![55.0, 66.0]),
            )])
        );
    }

    #[test]
    fn network_gradients_add_rejects_incompatible_layer_counts() {
        let left = NetworkGradients::new(vec![sample_layer_gradient()]);
        let right = NetworkGradients::new(vec![]);

        assert_eq!(
            left.add(&right),
            Err(NetworkError::GradientLayerCountMismatch { left: 1, right: 0 })
        );
    }

    #[test]
    fn network_gradients_add_propagates_layer_gradient_errors() {
        let left = NetworkGradients::new(vec![LayerGradient::new(
            vec![Vector::new(vec![1.0, 2.0])],
            Vector::new(vec![3.0]),
        )]);
        let right = NetworkGradients::new(vec![LayerGradient::new(
            vec![Vector::new(vec![4.0])],
            Vector::new(vec![5.0]),
        )]);

        assert_eq!(
            left.add(&right),
            Err(NetworkError::Linalg(LinalgError::DimensionMismatch {
                left: 2,
                right: 1,
            }))
        );
    }

    #[test]
    fn network_gradients_scale_every_layer() {
        let gradients = NetworkGradients::new(vec![
            sample_layer_gradient(),
            LayerGradient::new(vec![Vector::new(vec![7.0])], Vector::new(vec![8.0])),
        ]);

        let result = gradients.scale(0.5);

        assert_eq!(
            result,
            NetworkGradients::new(vec![
                LayerGradient::new(
                    vec![Vector::new(vec![0.5, 1.0]), Vector::new(vec![1.5, 2.0])],
                    Vector::new(vec![2.5, 3.0]),
                ),
                LayerGradient::new(vec![Vector::new(vec![3.5])], Vector::new(vec![4.0])),
            ])
        );
    }

    #[test]
    fn network_gradients_scale_by_one_preserves_values() {
        let gradients = NetworkGradients::new(vec![sample_layer_gradient()]);

        assert_eq!(gradients.scale(1.0), gradients);
    }

    #[test]
    fn network_gradients_scale_by_zero_zeros_every_value() {
        let gradients = NetworkGradients::new(vec![
            sample_layer_gradient(),
            LayerGradient::new(vec![Vector::new(vec![7.0])], Vector::new(vec![8.0])),
        ]);

        assert_eq!(
            gradients.scale(0.0),
            NetworkGradients::new(vec![
                LayerGradient::new(
                    vec![Vector::new(vec![0.0, 0.0]), Vector::new(vec![0.0, 0.0])],
                    Vector::new(vec![0.0, 0.0]),
                ),
                LayerGradient::new(vec![Vector::new(vec![0.0])], Vector::new(vec![0.0])),
            ])
        );
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

    #[test]
    fn backpropagate_mse_computes_single_layer_gradients() {
        let network = Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![0.5, -1.0]], vec![0.25]),
            Activation::Linear,
        )])
        .unwrap();

        let gradients = network
            .backpropagate_mse(&Vector::new(vec![2.0, -3.0]), &Vector::new(vec![1.0]))
            .unwrap();

        let layer_gradient = &gradients.layer_gradients()[0];

        assert_eq!(
            layer_gradient.weight_gradients(),
            &[Vector::new(vec![13.0, -19.5])]
        );
        assert_eq!(layer_gradient.bias_gradients(), &Vector::new(vec![6.5]));
    }

    #[test]
    fn backpropagate_mse_uses_chain_rule_for_hidden_layers() {
        let network = Network::new(vec![
            NetworkLayer::new(
                dense_layer(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0.0, 0.0]),
                Activation::Linear,
            ),
            NetworkLayer::new(
                dense_layer(vec![vec![2.0, -1.0]], vec![0.0]),
                Activation::Linear,
            ),
        ])
        .unwrap();

        let gradients = network
            .backpropagate_mse(&Vector::new(vec![1.0, 3.0]), &Vector::new(vec![1.0]))
            .unwrap();

        assert_eq!(
            gradients.layer_gradients()[0].weight_gradients(),
            &[Vector::new(vec![-8.0, -24.0]), Vector::new(vec![4.0, 12.0])]
        );
        assert_eq!(
            gradients.layer_gradients()[0].bias_gradients(),
            &Vector::new(vec![-8.0, 4.0])
        );
        assert_eq!(
            gradients.layer_gradients()[1].weight_gradients(),
            &[Vector::new(vec![-4.0, -12.0])]
        );
        assert_eq!(
            gradients.layer_gradients()[1].bias_gradients(),
            &Vector::new(vec![-4.0])
        );
    }

    #[test]
    fn backpropagate_mse_rejects_target_dimension_mismatch() {
        let network = Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![1.0, 1.0]], vec![0.0]),
            Activation::Linear,
        )])
        .unwrap();

        let result =
            network.backpropagate_mse(&Vector::new(vec![1.0, 2.0]), &Vector::new(vec![1.0, 2.0]));

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
    fn backpropagate_mse_rejects_softmax_activation() {
        let network = Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0.0, 0.0]),
            Activation::Softmax,
        )])
        .unwrap();

        let result =
            network.backpropagate_mse(&Vector::new(vec![1.0, 2.0]), &Vector::new(vec![0.0, 1.0]));

        assert_eq!(
            result,
            Err(NetworkError::UnsupportedActivation {
                layer_index: 0,
                activation: Activation::Softmax
            })
        );
    }

    #[test]
    fn softmax_cross_entropy_uses_fused_prediction_minus_target_gradient() {
        let network = Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![0.0], vec![0.0], vec![0.0]], vec![0.0, 0.0, 0.0]),
            Activation::Softmax,
        )])
        .unwrap();
        let gradients = network
            .backpropagate_softmax_cross_entropy(
                &Vector::new(vec![2.0]),
                &Vector::new(vec![1.0, 0.0, 0.0]),
            )
            .unwrap();
        let layer = &gradients.layer_gradients()[0];
        assert_close(layer.bias_gradients()[0], -2.0 / 3.0);
        assert_close(layer.bias_gradients()[1], 1.0 / 3.0);
        assert_close(layer.bias_gradients()[2], 1.0 / 3.0);
        assert_close(layer.weight_gradients()[0][0], -4.0 / 3.0);
        assert_close(layer.weight_gradients()[1][0], 2.0 / 3.0);
    }

    #[test]
    fn apply_gradients_updates_network_weights() {
        let mut network = Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![1.0, -2.0]], vec![0.5]),
            Activation::Linear,
        )])
        .unwrap();
        let gradients = NetworkGradients::new(vec![LayerGradient::new(
            vec![Vector::new(vec![0.25, -0.5])],
            Vector::new(vec![1.0]),
        )]);

        network.apply_gradients(&gradients, 0.1).unwrap();

        let neuron = &network.layers()[0].layer().neurons()[0];

        assert_eq!(neuron.weights(), &Vector::new(vec![0.975, -1.95]));
        assert_eq!(neuron.bias(), 0.4);
    }

    #[test]
    fn apply_gradients_rejects_invalid_learning_rate() {
        let mut network = Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![1.0]], vec![0.0]),
            Activation::Linear,
        )])
        .unwrap();
        let gradients = NetworkGradients::new(vec![LayerGradient::new(
            vec![Vector::new(vec![1.0])],
            Vector::new(vec![1.0]),
        )]);

        let result = network.apply_gradients(&gradients, 0.0);

        assert_eq!(
            result,
            Err(NetworkError::InvalidLearningRate { value: 0.0 })
        );
    }

    #[test]
    fn train_mse_updates_weights_and_reduces_loss() {
        let mut network = Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![0.0]], vec![0.0]),
            Activation::Linear,
        )])
        .unwrap();
        let input = Vector::new(vec![1.0]);
        let target = Vector::new(vec![1.0]);
        let initial_prediction = network.forward(&input).unwrap();
        let initial_error = initial_prediction[0] - target[0];

        let gradients = network.train_mse(&input, &target, 0.1).unwrap();

        let updated_prediction = network.forward(&input).unwrap();
        let updated_error = updated_prediction[0] - target[0];

        assert_eq!(
            gradients.layer_gradients()[0].weight_gradients(),
            &[Vector::new(vec![-2.0])]
        );
        assert!(updated_error.abs() < initial_error.abs());
        assert_close(updated_prediction[0], 0.4);
    }
}
