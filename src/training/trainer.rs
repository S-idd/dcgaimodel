use crate::dataset::{Dataset, DatasetBatch, DatasetError};
use crate::linalg::Vector;
use crate::nn::{Network, NetworkError, NetworkGradients, Optimizer, OptimizerError};
use std::error::Error;
use std::fmt;

/// Errors that can occur while training a neural network.
#[derive(Debug, PartialEq)]
pub enum TrainerError {
    /// The training dataset is empty.
    EmptyDataset,

    /// The configured number of epochs is invalid.
    InvalidEpochs { epochs: usize },

    /// A dataset operation failed.
    Dataset(DatasetError),

    /// A network operation failed.
    Network(NetworkError),

    /// An optimizer operation failed.
    Optimizer(OptimizerError),
}

impl fmt::Display for TrainerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainerError::EmptyDataset => {
                write!(f, "Cannot train on an empty dataset.")
            }
            TrainerError::InvalidEpochs { epochs } => {
                write!(f, "Number of epochs must be at least 1: {}", epochs)
            }
            TrainerError::Dataset(error) => {
                write!(f, "Dataset error: {}", error)
            }
            TrainerError::Network(error) => {
                write!(f, "Network error: {}", error)
            }
            TrainerError::Optimizer(error) => {
                write!(f, "Optimizer error: {}", error)
            }
        }
    }
}

impl Error for TrainerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            TrainerError::Dataset(error) => Some(error),
            TrainerError::Network(error) => Some(error),
            TrainerError::Optimizer(error) => Some(error),
            TrainerError::EmptyDataset | TrainerError::InvalidEpochs { .. } => None,
        }
    }
}

impl From<DatasetError> for TrainerError {
    fn from(error: DatasetError) -> Self {
        Self::Dataset(error)
    }
}

impl From<NetworkError> for TrainerError {
    fn from(error: NetworkError) -> Self {
        Self::Network(error)
    }
}

impl From<OptimizerError> for TrainerError {
    fn from(error: OptimizerError) -> Self {
        Self::Optimizer(error)
    }
}

/// Configuration used by the training loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingConfig {
    /// Number of complete passes over the training dataset.
    pub epochs: usize,

    /// Number of examples processed before one optimizer update.
    pub batch_size: usize,
}

impl TrainingConfig {
    /// Creates a training configuration.
    pub fn new(epochs: usize, batch_size: usize) -> Result<Self, TrainerError> {
        if epochs == 0 {
            return Err(TrainerError::InvalidEpochs { epochs });
        }

        if batch_size == 0 {
            return Err(TrainerError::Dataset(DatasetError::InvalidBatchSize {
                batch_size,
            }));
        }

        Ok(Self { epochs, batch_size })
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
        }
    }
}

/// Loss information collected during training.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingHistory {
    epoch_losses: Vec<f64>,
}

impl TrainingHistory {
    fn new() -> Self {
        Self {
            epoch_losses: Vec::new(),
        }
    }

    /// Returns the average loss recorded for every epoch.
    pub fn epoch_losses(&self) -> &[f64] {
        &self.epoch_losses
    }

    /// Returns the number of completed epochs.
    pub fn len(&self) -> usize {
        self.epoch_losses.len()
    }

    /// Returns true when no epochs have been recorded.
    pub fn is_empty(&self) -> bool {
        self.epoch_losses.is_empty()
    }

    /// Returns the loss for a particular epoch.
    pub fn get(&self, epoch: usize) -> Option<f64> {
        self.epoch_losses.get(epoch).copied()
    }

    fn push(&mut self, loss: f64) {
        self.epoch_losses.push(loss);
    }
}

/// Coordinates dataset iteration, backpropagation, and optimization.
///
/// `Trainer` owns the network and optimizer and performs mini-batch
/// gradient-descent training using the network's existing MSE
/// backpropagation implementation.
#[derive(Debug)]
pub struct Trainer<O> {
    network: Network,
    optimizer: O,
    config: TrainingConfig,
}

impl<O: Optimizer> Trainer<O> {
    /// Creates a trainer from a network, optimizer, and training configuration.
    pub fn new(network: Network, optimizer: O, config: TrainingConfig) -> Self {
        Self {
            network,
            optimizer,
            config,
        }
    }

    /// Returns an immutable reference to the network.
    pub fn network(&self) -> &Network {
        &self.network
    }

    /// Returns mutable access to the network.
    pub fn network_mut(&mut self) -> &mut Network {
        &mut self.network
    }

    /// Returns an immutable reference to the optimizer.
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Returns mutable access to the optimizer.
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Returns the current training configuration.
    pub fn config(&self) -> TrainingConfig {
        self.config
    }

    /// Replaces the training configuration.
    pub fn set_config(&mut self, config: TrainingConfig) {
        self.config = config;
    }

    /// Trains the network for the configured number of epochs.
    ///
    /// Each epoch:
    /// 1. Splits the dataset into deterministic mini-batches.
    /// 2. Computes an MSE gradient for every example in a batch.
    /// 3. Averages the gradients across the batch.
    /// 4. Passes the averaged gradients to the optimizer.
    /// 5. Records the average pre-update batch loss.
    pub fn train(&mut self, dataset: &Dataset) -> Result<TrainingHistory, TrainerError> {
        if dataset.is_empty() {
            return Err(TrainerError::EmptyDataset);
        }

        let mut history = TrainingHistory::new();

        for _epoch in 0..self.config.epochs {
            let batches = dataset.batches(self.config.batch_size)?;

            let mut epoch_loss = 0.0;
            let mut epoch_examples = 0usize;

            for batch in &batches {
                let (gradients, batch_loss) = self.compute_batch_gradients(batch)?;

                self.optimizer.step(&mut self.network, &gradients)?;

                epoch_loss += batch_loss * batch.len() as f64;
                epoch_examples += batch.len();
            }

            let average_epoch_loss = epoch_loss / epoch_examples as f64;
            history.push(average_epoch_loss);
        }

        Ok(history)
    }

    /// Computes the average MSE gradients for one mini-batch.
    ///
    /// The loss values are evaluated before the optimizer update.
    fn compute_batch_gradients(
        &self,
        batch: &DatasetBatch,
    ) -> Result<(NetworkGradients, f64), TrainerError> {
        if batch.is_empty() {
            return Err(TrainerError::EmptyDataset);
        }

        if batch.features().len() != batch.targets().len() {
            return Err(TrainerError::Dataset(DatasetError::RowCountMismatch {
                features: batch.features().len(),
                targets: batch.targets().len(),
            }));
        }

        let mut accumulated_gradients: Option<NetworkGradients> = None;
        let mut total_loss = 0.0;

        for (inputs, target) in batch.features().iter().zip(batch.targets().iter()) {
            let prediction = self.network.forward(inputs)?;

            if prediction.len() != target.len() {
                return Err(TrainerError::Network(NetworkError::DimensionMismatch {
                    layer_index: self.network.len(),
                    expected: prediction.len(),
                    actual: target.len(),
                }));
            }

            let sample_loss = mean_squared_error(&prediction, target);
            total_loss += sample_loss;

            let gradients = self.network.backpropagate_mse(inputs, target)?;

            if let Some(accumulated) = accumulated_gradients.take() {
                accumulated_gradients = Some(accumulated.add(&gradients)?);
            } else {
                accumulated_gradients = Some(gradients);
            }
        }

        let gradients = accumulated_gradients
            .ok_or(TrainerError::EmptyDataset)?
            .scale(1.0 / batch.len() as f64);

        Ok((gradients, total_loss / batch.len() as f64))
    }
}

/// Computes MSE directly from two vectors.
///
/// This helper intentionally mirrors the MSE calculation already used by
/// `Network::backpropagate_mse`, keeping the trainer independent from the
/// concrete `MeanSquaredError` type.
fn mean_squared_error(predicted: &Vector, target: &Vector) -> f64 {
    predicted
        .iter()
        .zip(target.iter())
        .map(|(prediction, expected)| {
            let error = prediction - expected;
            error * error
        })
        .sum::<f64>()
        / predicted.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::linalg::Vector;
    use crate::nn::{Layer, NetworkLayer};

    struct TestOptimizer;

    impl Optimizer for TestOptimizer {
        fn step(
            &mut self,
            network: &mut Network,
            gradients: &NetworkGradients,
        ) -> Result<(), OptimizerError> {
            network
                .apply_gradients(gradients, 0.1)
                .map_err(optimizer_error_from_network)
        }
    }

    fn optimizer_error_from_network(error: NetworkError) -> OptimizerError {
        OptimizerError::Network(error)
    }

    fn dense_layer(weights: Vec<Vec<f64>>, biases: Vec<f64>) -> Layer {
        let weights = weights.into_iter().map(Vector::new).collect();

        Layer::dense(weights, biases).unwrap()
    }

    fn single_neuron_network(weight: f64, bias: f64) -> Network {
        Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![weight]], vec![bias]),
            Activation::Linear,
        )])
        .unwrap()
    }

    #[test]
    fn creates_training_config() {
        let config = TrainingConfig::new(5, 4).unwrap();

        assert_eq!(
            config,
            TrainingConfig {
                epochs: 5,
                batch_size: 4
            }
        );
    }

    #[test]
    fn rejects_zero_epochs() {
        let result = TrainingConfig::new(0, 4);

        assert!(matches!(
            result,
            Err(TrainerError::InvalidEpochs { epochs: 0 })
        ));
    }

    #[test]
    fn rejects_zero_batch_size() {
        let result = TrainingConfig::new(5, 0);

        assert!(matches!(
            result,
            Err(TrainerError::Dataset(DatasetError::InvalidBatchSize {
                batch_size: 0
            }))
        ));
    }

    #[test]
    fn default_training_config_is_valid() {
        let config = TrainingConfig::default();

        assert_eq!(config.epochs, 10);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn creates_trainer() {
        let network = single_neuron_network(0.0, 0.0);
        let optimizer = TestOptimizer;
        let config = TrainingConfig::new(2, 1).unwrap();

        let trainer = Trainer::new(network, optimizer, config);

        assert_eq!(trainer.network().len(), 1);
        assert_eq!(trainer.config(), config);
    }

    #[test]
    fn rejects_empty_dataset() {
        let network = single_neuron_network(0.0, 0.0);
        let optimizer = TestOptimizer;
        let config = TrainingConfig::new(1, 1).unwrap();

        let mut trainer = Trainer::new(network, optimizer, config);

        let dataset = Dataset::new(vec![], vec![]);

        assert_eq!(dataset, Err(DatasetError::EmptyDataset));

        /*
         * Since Dataset::new rejects an empty dataset, this test verifies the
         * dataset-layer contract rather than constructing an invalid Dataset.
         */
        let _ = &mut trainer;
    }

    #[test]
    fn computes_average_batch_gradients() {
        let network = single_neuron_network(0.0, 0.0);
        let optimizer = TestOptimizer;
        let config = TrainingConfig::new(1, 2).unwrap();

        let trainer = Trainer::new(network, optimizer, config);

        let dataset = Dataset::new(
            vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])],
            vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])],
        )
        .unwrap();

        let batch = dataset.batches(2).unwrap().remove(0);

        let (gradients, loss) = trainer.compute_batch_gradients(&batch).unwrap();

        assert!((loss - 2.5).abs() < 1e-10);

        /*
         * Sample 1 gradient:
         *   dL/dw = -2
         *
         * Sample 2 gradient:
         *   dL/dw = -8
         *
         * Average:
         *   (-2 + -8) / 2 = -5
         */
        assert_eq!(
            gradients.layer_gradients()[0].weight_gradients(),
            &[Vector::new(vec![-5.0])]
        );

        assert_eq!(
            gradients.layer_gradients()[0].bias_gradients(),
            &Vector::new(vec![-3.0])
        );
    }

    #[test]
    fn trains_for_configured_number_of_epochs() {
        let network = single_neuron_network(0.0, 0.0);
        let optimizer = TestOptimizer;
        let config = TrainingConfig::new(3, 1).unwrap();

        let mut trainer = Trainer::new(network, optimizer, config);

        let dataset =
            Dataset::new(vec![Vector::new(vec![1.0])], vec![Vector::new(vec![1.0])]).unwrap();

        let history = trainer.train(&dataset).unwrap();

        assert_eq!(history.len(), 3);
        assert_eq!(history.epoch_losses().len(), 3);
    }

    #[test]
    fn training_reduces_loss() {
        let network = single_neuron_network(0.0, 0.0);
        let optimizer = TestOptimizer;
        let config = TrainingConfig::new(5, 1).unwrap();

        let mut trainer = Trainer::new(network, optimizer, config);

        let dataset =
            Dataset::new(vec![Vector::new(vec![1.0])], vec![Vector::new(vec![1.0])]).unwrap();

        let history = trainer.train(&dataset).unwrap();

        assert!(history.epoch_losses()[1] < history.epoch_losses()[0]);
        assert!(history.epoch_losses()[4] < history.epoch_losses()[0]);
    }

    #[test]
    fn batch_size_larger_than_dataset_is_supported() {
        let network = single_neuron_network(0.0, 0.0);
        let optimizer = TestOptimizer;
        let config = TrainingConfig::new(1, 100).unwrap();

        let mut trainer = Trainer::new(network, optimizer, config);

        let dataset = Dataset::new(
            vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])],
            vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])],
        )
        .unwrap();

        let history = trainer.train(&dataset).unwrap();

        assert_eq!(history.len(), 1);
        assert!(history.epoch_losses()[0].is_finite());
    }

    #[test]
    fn history_get_returns_epoch_loss() {
        let history = TrainingHistory {
            epoch_losses: vec![1.0, 0.5, 0.25],
        };

        assert_eq!(history.get(0), Some(1.0));
        assert_eq!(history.get(1), Some(0.5));
        assert_eq!(history.get(2), Some(0.25));
        assert_eq!(history.get(3), None);
    }

    #[test]
    fn records_an_example_weighted_epoch_loss_for_partial_batches() {
        let network = single_neuron_network(0.0, 0.0);
        let optimizer = TestOptimizer;
        let config = TrainingConfig::new(1, 2).unwrap();
        let mut trainer = Trainer::new(network, optimizer, config);
        let dataset = Dataset::new(
            vec![
                Vector::new(vec![1.0]),
                Vector::new(vec![2.0]),
                Vector::new(vec![3.0]),
            ],
            vec![
                Vector::new(vec![1.0]),
                Vector::new(vec![3.0]),
                Vector::new(vec![2.0]),
            ],
        )
        .unwrap();

        let history = trainer.train(&dataset).unwrap();

        // Batch losses before their respective updates are 5.0 (two examples)
        // and 0.25 (one example), so the epoch average is (10.0 + 0.25) / 3.
        assert!((history.epoch_losses()[0] - 10.25 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn rejects_batch_with_mismatched_feature_and_target_counts() {
        let trainer = Trainer::new(
            single_neuron_network(0.0, 0.0),
            TestOptimizer,
            TrainingConfig::new(1, 1).unwrap(),
        );
        let batch = DatasetBatch::new(
            vec![Vector::new(vec![1.0])],
            vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])],
        );

        assert_eq!(
            trainer.compute_batch_gradients(&batch),
            Err(TrainerError::Dataset(DatasetError::RowCountMismatch {
                features: 1,
                targets: 2,
            }))
        );
    }
}
