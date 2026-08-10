use super::error::OptimizerError;
use super::traits::Optimizer;
use super::utils::{
    combine_gradients, ensure_state_shape, validate_learning_rate, validate_momentum,
};
use crate::linalg::Vector;
use crate::nn::{Network, NetworkGradients};

/// Stochastic Gradient Descent with momentum.
#[derive(Debug, Clone, PartialEq)]
pub struct Momentum {
    learning_rate: f64,
    momentum: f64,
    velocity: Option<NetworkGradients>,
}

impl Momentum {
    /// Creates a momentum optimizer.
    pub fn new(learning_rate: f64, momentum: f64) -> Result<Self, OptimizerError> {
        validate_learning_rate(learning_rate)?;
        validate_momentum(momentum)?;

        Ok(Self {
            learning_rate,
            momentum,
            velocity: None,
        })
    }

    /// Returns the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the current momentum coefficient.
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// Updates the learning rate.
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<(), OptimizerError> {
        validate_learning_rate(learning_rate)?;

        self.learning_rate = learning_rate;

        Ok(())
    }

    /// Updates the momentum coefficient.
    pub fn set_momentum(&mut self, momentum: f64) -> Result<(), OptimizerError> {
        validate_momentum(momentum)?;

        self.momentum = momentum;

        Ok(())
    }

    /// Returns the current velocity state.
    pub fn velocity(&self) -> Option<&NetworkGradients> {
        self.velocity.as_ref()
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

impl Optimizer for Momentum {
    fn step(
        &mut self,
        network: &mut Network,
        gradients: &NetworkGradients,
    ) -> Result<(), OptimizerError> {
        ensure_state_shape(&mut self.velocity, gradients);

        let velocity = self
            .velocity
            .as_ref()
            .expect("velocity state is initialized before use");
        let updated_velocity = combine_gradients(velocity, gradients, |previous, gradient| {
            self.momentum * previous + gradient
        });

        network.apply_gradients(&updated_velocity, self.learning_rate)?;
        self.velocity = Some(updated_velocity);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::LayerGradient;
    use crate::nn::optimizer::test_support::{assert_close, single_neuron_network};

    #[test]
    fn creates_momentum_optimizer() {
        let optimizer = Momentum::new(0.01, 0.9).unwrap();

        assert_eq!(optimizer.learning_rate(), 0.01);
        assert_eq!(optimizer.momentum(), 0.9);
        assert!(optimizer.velocity().is_none());
    }

    #[test]
    fn rejects_invalid_momentum() {
        assert_eq!(
            Momentum::new(0.01, 1.0),
            Err(OptimizerError::InvalidMomentum { value: 1.0 })
        );
        assert_eq!(
            Momentum::new(0.01, -0.1),
            Err(OptimizerError::InvalidMomentum { value: -0.1 })
        );
    }

    #[test]
    fn updates_momentum_learning_rate_and_coefficient() {
        let mut optimizer = Momentum::new(0.01, 0.8).unwrap();

        optimizer.set_learning_rate(0.2).unwrap();
        optimizer.set_momentum(0.95).unwrap();

        assert_eq!(optimizer.learning_rate(), 0.2);
        assert_eq!(optimizer.momentum(), 0.95);
    }

    #[test]
    fn momentum_step_accumulates_velocity() {
        let mut network = single_neuron_network(1.0, 0.5);
        let mut optimizer = Momentum::new(0.1, 0.9).unwrap();
        let gradients = NetworkGradients::new(vec![LayerGradient::new(
            vec![Vector::new(vec![0.25])],
            Vector::new(vec![1.0]),
        )]);

        optimizer.step(&mut network, &gradients).unwrap();
        optimizer.step(&mut network, &gradients).unwrap();

        let neuron = &network.layers()[0].layer().neurons()[0];
        let velocity = optimizer.velocity().unwrap();

        assert_close(neuron.weights()[0], 0.9275);
        assert_close(neuron.bias(), 0.21);
        assert_eq!(
            velocity.layer_gradients()[0].weight_gradients(),
            &[Vector::new(vec![0.475])]
        );
        assert_eq!(
            velocity.layer_gradients()[0].bias_gradients(),
            &Vector::new(vec![1.9])
        );
    }

    #[test]
    fn momentum_train_mse_step_updates_weights_and_reduces_loss() {
        let mut network = single_neuron_network(0.0, 0.0);
        let mut optimizer = Momentum::new(0.1, 0.9).unwrap();
        let input = Vector::new(vec![1.0]);
        let target = Vector::new(vec![1.0]);
        let initial_prediction = network.forward(&input).unwrap();
        let initial_error = initial_prediction[0] - target[0];

        optimizer
            .train_mse_step(&mut network, &input, &target)
            .unwrap();

        let updated_prediction = network.forward(&input).unwrap();
        let updated_error = updated_prediction[0] - target[0];

        assert!(updated_error.abs() < initial_error.abs());
        assert_close(updated_prediction[0], 0.4);
    }
}
