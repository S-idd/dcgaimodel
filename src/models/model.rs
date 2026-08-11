use crate::activations::Activation;
use crate::dataset::Dataset;
use crate::linalg::{LinalgError, Vector};
use crate::nn::{Layer, Network, NetworkError, NetworkLayer, Optimizer};
use crate::prediction::{ModelPrediction, PredictionError, PredictionKind, Predictor};
use crate::training::{Trainer, TrainerError, TrainingConfig, TrainingHistory};
use std::error::Error;
use std::fmt;

/// Architecture and output-interpretation configuration for a DCG model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    /// Input feature count; `ContractFeatures` currently emits eight values.
    pub input_size: usize,
    /// Positive neuron counts for optional hidden ReLU layers.
    pub hidden_layer_sizes: Vec<usize>,
    /// Classification decision boundary. `0.5` is the default.
    pub threshold: f64,
}

impl ModelConfig {
    /// Creates a configuration after validating dimensions and threshold.
    pub fn new(
        input_size: usize,
        hidden_layer_sizes: Vec<usize>,
        threshold: f64,
    ) -> Result<Self, ModelError> {
        if input_size == 0 {
            return Err(ModelError::InvalidLayerSize { value: input_size });
        }
        if let Some(&value) = hidden_layer_sizes.iter().find(|&&size| size == 0) {
            return Err(ModelError::InvalidLayerSize { value });
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(ModelError::Prediction(PredictionError::InvalidThreshold {
                value: threshold,
            }));
        }
        Ok(Self {
            input_size,
            hidden_layer_sizes,
            threshold,
        })
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_size: 8,
            hidden_layer_sizes: vec![6],
            threshold: 0.5,
        }
    }
}

/// Errors returned by model construction, training, and prediction.
#[derive(Debug)]
pub enum ModelError {
    /// A model dimension was zero.
    InvalidLayerSize { value: usize },
    /// A wrapped network did not meet the single-output model contract.
    OutputDimensionMismatch { expected: usize, actual: usize },
    /// Network construction failed.
    Network(NetworkError),
    /// Dense-layer construction failed.
    Linalg(LinalgError),
    /// Training failed through the existing trainer.
    Training(TrainerError),
    /// Prediction validation failed.
    Prediction(PredictionError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLayerSize { value } => {
                write!(f, "Model layer size must be positive: {value}")
            }
            Self::OutputDimensionMismatch { expected, actual } => write!(
                f,
                "Model output dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::Network(error) => write!(f, "Network construction error: {error}"),
            Self::Linalg(error) => write!(f, "Layer construction error: {error}"),
            Self::Training(error) => write!(f, "Training error: {error}"),
            Self::Prediction(error) => write!(f, "Prediction error: {error}"),
        }
    }
}

impl Error for ModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Network(error) => Some(error),
            Self::Linalg(error) => Some(error),
            Self::Training(error) => Some(error),
            Self::Prediction(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NetworkError> for ModelError {
    fn from(error: NetworkError) -> Self {
        Self::Network(error)
    }
}
impl From<LinalgError> for ModelError {
    fn from(error: LinalgError) -> Self {
        Self::Linalg(error)
    }
}
impl From<TrainerError> for ModelError {
    fn from(error: TrainerError) -> Self {
        Self::Training(error)
    }
}
impl From<PredictionError> for ModelError {
    fn from(error: PredictionError) -> Self {
        Self::Prediction(error)
    }
}

/// A task-specific wrapper around the shared generic [`Network`].
#[derive(Debug, Clone)]
pub struct DcgModel {
    kind: PredictionKind,
    network: Network,
    threshold: f64,
}

impl DcgModel {
    /// Wraps a single-output network for a DCG prediction task.
    pub fn new(kind: PredictionKind, network: Network, threshold: f64) -> Result<Self, ModelError> {
        if network.output_size() != 1 {
            return Err(ModelError::OutputDimensionMismatch {
                expected: 1,
                actual: network.output_size(),
            });
        }
        Predictor::new(&network, kind, threshold)?;
        Ok(Self {
            kind,
            network,
            threshold,
        })
    }

    /// Creates a deterministic, small shared network with sigmoid output.
    pub fn with_default_network(
        kind: PredictionKind,
        config: ModelConfig,
    ) -> Result<Self, ModelError> {
        let network = build_network(&config)?;
        Self::new(kind, network, config.threshold)
    }

    /// Returns the model task interpretation.
    pub fn kind(&self) -> PredictionKind {
        self.kind
    }
    /// Returns the generic trained network.
    pub fn network(&self) -> &Network {
        &self.network
    }
    /// Returns the classification threshold (unused for risk interpretation).
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Predicts one input through the network and task-specific output adapter.
    pub fn predict(&self, features: &Vector) -> Result<ModelPrediction, ModelError> {
        Ok(Predictor::new(&self.network, self.kind, self.threshold)?.predict(features)?)
    }

    /// Predicts a batch through the network and task-specific output adapter.
    pub fn predict_batch(&self, features: &[Vector]) -> Result<Vec<ModelPrediction>, ModelError> {
        Ok(Predictor::new(&self.network, self.kind, self.threshold)?.predict_batch(features)?)
    }

    /// Trains this model using the existing trainer and a caller-selected optimizer.
    pub fn train<O: Optimizer>(
        &mut self,
        dataset: &Dataset,
        optimizer: O,
        config: TrainingConfig,
    ) -> Result<TrainingHistory, ModelError> {
        let mut trainer = Trainer::new(self.network.clone(), optimizer, config);
        let history = trainer.train(dataset)?;
        self.network = trainer.network().clone();
        Ok(history)
    }
}

fn build_network(config: &ModelConfig) -> Result<Network, ModelError> {
    let mut dimensions = Vec::with_capacity(config.hidden_layer_sizes.len() + 2);
    dimensions.push(config.input_size);
    dimensions.extend(config.hidden_layer_sizes.iter().copied());
    dimensions.push(1);
    let layers = dimensions
        .windows(2)
        .enumerate()
        .map(|(layer_index, sizes)| {
            let activation = if layer_index + 1 == dimensions.len() - 1 {
                Activation::Sigmoid
            } else {
                Activation::Relu
            };
            deterministic_layer(sizes[0], sizes[1], layer_index, activation)
        })
        .collect::<Result<Vec<_>, ModelError>>()?;
    Ok(Network::new(layers)?)
}

fn deterministic_layer(
    input_size: usize,
    output_size: usize,
    layer_index: usize,
    activation: Activation,
) -> Result<NetworkLayer, ModelError> {
    let weights = (0..output_size)
        .map(|neuron_index| {
            Vector::new(
                (0..input_size)
                    .map(|input_index| {
                        let position = (layer_index + 1) * (neuron_index + 2) * (input_index + 3);
                        (position % 11) as f64 / 100.0 - 0.05
                    })
                    .collect(),
            )
        })
        .collect();
    let biases = (0..output_size)
        .map(|neuron_index| ((layer_index + neuron_index) % 5) as f64 / 100.0 - 0.02)
        .collect();
    Ok(NetworkLayer::new(
        Layer::dense(weights, biases)?,
        activation,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{build_dataset, synthetic_examples};
    use crate::nn::Sgd;

    #[test]
    fn creates_and_trains_shared_model() {
        let mut model =
            DcgModel::with_default_network(PredictionKind::BreakingChange, ModelConfig::default())
                .unwrap();
        let history = model
            .train(
                &build_dataset(&synthetic_examples(), PredictionKind::BreakingChange).unwrap(),
                Sgd::new(0.05).unwrap(),
                TrainingConfig::new(3, 2).unwrap(),
            )
            .unwrap();
        assert_eq!(history.len(), 3);
        assert!(history.epoch_losses().iter().all(|loss| loss.is_finite()));
    }

    #[test]
    fn rejects_invalid_model_configuration() {
        assert!(matches!(
            ModelConfig::new(0, vec![], 0.5),
            Err(ModelError::InvalidLayerSize { value: 0 })
        ));
        assert!(matches!(
            ModelConfig::new(8, vec![0], 0.5),
            Err(ModelError::InvalidLayerSize { value: 0 })
        ));
    }
}
