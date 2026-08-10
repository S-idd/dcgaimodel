use super::error::OptimizerError;
use super::traits::Optimizer;
use super::utils::validate_learning_rate;
use crate::linalg::Vector;
use crate::nn::{Network, NetworkGradients};

/// Stochastic Gradient Descent optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct Sgd {
    learning_rate: f64,
}

impl Sgd {
    /// Creates an SGD optimizer.
    pub fn new(learning_rate: f64) -> Result<Self, OptimizerError> {
        validate_learning_rate(learning_rate)?;

        Ok(Self { learning_rate })
    }

    /// Returns the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Updates the learning rate.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<(), OptimizerError> {
        validate_learning_rate(learning_rate)?;

        self.learning_rate = learning_rate;

        Ok(())
    }

    /// Runs one MSE training step using this optimizer.
    pub fn train_mse_step(
        &mut self,
        network: &mut Network,
        inputs: &Vector,
        target: &Vector,
    ) -> Result<NetworkGradients, OptimizerError> {
        let gradients = network.backpropagate_mse(inputs, target)?;

        self.step(network, &gradients)?;

        Ok(gradients)
    }
}

impl Optimizer for Sgd {
    fn step(
        &mut self,
        network: &mut Network,
        gradients: &NetworkGradients,
    ) -> Result<(), OptimizerError> {
        network.apply_gradients(gradients, self.learning_rate)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::optimizer::test_support::single_neuron_network;
    use crate::nn::{LayerGradient, NetworkError};

    #[test]
    fn creates_sgd_optimizer() {
        let optimizer = Sgd::new(0.01).unwrap();

        assert_eq!(optimizer.learning_rate(), 0.01);
    }

    #[test]
    fn rejects_invalid_learning_rate() {
        assert_eq!(
            Sgd::new(0.0),
            Err(OptimizerError::InvalidLearningRate { value: 0.0 })
        );
        assert_eq!(
            Sgd::new(f64::INFINITY),
            Err(OptimizerError::InvalidLearningRate {
                value: f64::INFINITY
            })
        );
    }

    #[test]
    fn updates_learning_rate() {
        let mut optimizer = Sgd::new(0.01).unwrap();

        optimizer.set_learning_rate(0.2).unwrap();

        assert_eq!(optimizer.learning_rate(), 0.2);
    }

    #[test]
    fn step_applies_gradients_to_network() {
        let mut network = single_neuron_network(1.0, 0.5);
        let mut optimizer = Sgd::new(0.1).unwrap();
        let gradients = NetworkGradients::new(vec![LayerGradient::new(
            vec![Vector::new(vec![0.25])],
            Vector::new(vec![1.0]),
        )]);

        optimizer.step(&mut network, &gradients).unwrap();

        let neuron = &network.layers()[0].layer().neurons()[0];

        assert_eq!(neuron.weights(), &Vector::new(vec![0.975]));
        assert_eq!(neuron.bias(), 0.4);
    }

    #[test]
    fn step_returns_network_errors() {
        let mut network = single_neuron_network(1.0, 0.0);
        let mut optimizer = Sgd::new(0.1).unwrap();
        let gradients = NetworkGradients::new(vec![]);

        let result = optimizer.step(&mut network, &gradients);

        assert_eq!(
            result,
            Err(OptimizerError::Network(NetworkError::DimensionMismatch {
                layer_index: 0,
                expected: 1,
                actual: 0
            }))
        );
    }

    #[test]
    fn train_mse_step_updates_weights_and_reduces_loss() {
        let mut network = single_neuron_network(0.0, 0.0);
        let mut optimizer = Sgd::new(0.1).unwrap();
        let input = Vector::new(vec![1.0]);
        let target = Vector::new(vec![1.0]);
        let initial_prediction = network.forward(&input).unwrap();
        let initial_error = initial_prediction[0] - target[0];

        let gradients = optimizer
            .train_mse_step(&mut network, &input, &target)
            .unwrap();

        let updated_prediction = network.forward(&input).unwrap();
        let updated_error = updated_prediction[0] - target[0];

        assert_eq!(
            gradients.layer_gradients()[0].weight_gradients(),
            &[Vector::new(vec![-2.0])]
        );
        assert!(updated_error.abs() < initial_error.abs());
        assert!((updated_prediction[0] - 0.4).abs() < 1e-10);
    }
}
