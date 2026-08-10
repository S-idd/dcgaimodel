use super::error::OptimizerError;
use super::traits::Optimizer;
use super::utils::{
    adam_update, combine_gradients, same_gradient_shape, validate_beta, validate_epsilon,
    validate_learning_rate, zeros_like,
};
use crate::linalg::Vector;
use crate::nn::{Network, NetworkGradients};

/// Adam optimizer.
#[derive(Debug, Clone, PartialEq)]
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    timestep: usize,
    first_moment: Option<NetworkGradients>,
    second_moment: Option<NetworkGradients>,
}

impl Adam {
    /// Creates an Adam optimizer using common default hyperparameters.
    pub fn new(learning_rate: f64) -> Result<Self, OptimizerError> {
        Self::with_hyperparameters(learning_rate, 0.9, 0.999, 1e-8)
    }

    /// Creates an Adam optimizer with explicit hyperparameters.
    pub fn with_hyperparameters(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Result<Self, OptimizerError> {
        validate_learning_rate(learning_rate)?;
        validate_beta(beta1)?;
        validate_beta(beta2)?;
        validate_epsilon(epsilon)?;

        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            timestep: 0,
            first_moment: None,
            second_moment: None,
        })
    }

    /// Returns the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the beta1 coefficient.
    pub fn beta1(&self) -> f64 {
        self.beta1
    }

    /// Returns the beta2 coefficient.
    pub fn beta2(&self) -> f64 {
        self.beta2
    }

    /// Returns the numerical stability epsilon.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Returns the number of optimizer steps already applied.
    pub fn timestep(&self) -> usize {
        self.timestep
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

impl Optimizer for Adam {
    fn step(
        &mut self,
        network: &mut Network,
        gradients: &NetworkGradients,
    ) -> Result<(), OptimizerError> {
        let state_reset = !self
            .first_moment
            .as_ref()
            .is_some_and(|state| same_gradient_shape(state, gradients))
            || !self
                .second_moment
                .as_ref()
                .is_some_and(|state| same_gradient_shape(state, gradients));

        if state_reset {
            self.first_moment = Some(zeros_like(gradients));
            self.second_moment = Some(zeros_like(gradients));
            self.timestep = 0;
        }

        let first_moment = self
            .first_moment
            .as_ref()
            .expect("first moment state is initialized before use");
        let second_moment = self
            .second_moment
            .as_ref()
            .expect("second moment state is initialized before use");

        let updated_first_moment =
            combine_gradients(first_moment, gradients, |moment, gradient| {
                self.beta1 * moment + (1.0 - self.beta1) * gradient
            });
        let updated_second_moment =
            combine_gradients(second_moment, gradients, |moment, gradient| {
                self.beta2 * moment + (1.0 - self.beta2) * gradient * gradient
            });

        self.timestep += 1;

        let update = adam_update(
            &updated_first_moment,
            &updated_second_moment,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.timestep,
        );

        network.apply_gradients(&update, self.learning_rate)?;
        self.first_moment = Some(updated_first_moment);
        self.second_moment = Some(updated_second_moment);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::LayerGradient;
    use crate::nn::optimizer::test_support::{assert_close, single_neuron_network};

    #[test]
    fn creates_adam_optimizer_with_defaults() {
        let optimizer = Adam::new(0.001).unwrap();

        assert_eq!(optimizer.learning_rate(), 0.001);
        assert_eq!(optimizer.beta1(), 0.9);
        assert_eq!(optimizer.beta2(), 0.999);
        assert_eq!(optimizer.epsilon(), 1e-8);
        assert_eq!(optimizer.timestep(), 0);
    }

    #[test]
    fn creates_adam_optimizer_with_hyperparameters() {
        let optimizer = Adam::with_hyperparameters(0.01, 0.8, 0.9, 1e-7).unwrap();

        assert_eq!(optimizer.learning_rate(), 0.01);
        assert_eq!(optimizer.beta1(), 0.8);
        assert_eq!(optimizer.beta2(), 0.9);
        assert_eq!(optimizer.epsilon(), 1e-7);
    }

    #[test]
    fn rejects_invalid_adam_hyperparameters() {
        assert_eq!(
            Adam::with_hyperparameters(0.0, 0.9, 0.999, 1e-8),
            Err(OptimizerError::InvalidLearningRate { value: 0.0 })
        );
        assert_eq!(
            Adam::with_hyperparameters(0.001, 1.0, 0.999, 1e-8),
            Err(OptimizerError::InvalidBeta { value: 1.0 })
        );
        assert_eq!(
            Adam::with_hyperparameters(0.001, 0.9, -0.1, 1e-8),
            Err(OptimizerError::InvalidBeta { value: -0.1 })
        );
        assert_eq!(
            Adam::with_hyperparameters(0.001, 0.9, 0.999, 0.0),
            Err(OptimizerError::InvalidEpsilon { value: 0.0 })
        );
    }

    #[test]
    fn updates_adam_learning_rate() {
        let mut optimizer = Adam::new(0.001).unwrap();

        optimizer.set_learning_rate(0.01).unwrap();

        assert_eq!(optimizer.learning_rate(), 0.01);
    }

    #[test]
    fn adam_step_applies_bias_corrected_update() {
        let mut network = single_neuron_network(1.0, 0.5);
        let mut optimizer = Adam::with_hyperparameters(0.1, 0.0, 0.0, 1e-12).unwrap();
        let gradients = NetworkGradients::new(vec![LayerGradient::new(
            vec![Vector::new(vec![0.25])],
            Vector::new(vec![1.0]),
        )]);

        optimizer.step(&mut network, &gradients).unwrap();

        let neuron = &network.layers()[0].layer().neurons()[0];

        assert_eq!(optimizer.timestep(), 1);
        assert_close(neuron.weights()[0], 0.9000000000004);
        assert_close(neuron.bias(), 0.4000000000001);
    }

    #[test]
    fn adam_train_mse_step_updates_weights_and_reduces_loss() {
        let mut network = single_neuron_network(0.0, 0.0);
        let mut optimizer = Adam::with_hyperparameters(0.1, 0.0, 0.0, 1e-12).unwrap();
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
        assert_close(updated_prediction[0], 0.1999999999999);
    }
}
