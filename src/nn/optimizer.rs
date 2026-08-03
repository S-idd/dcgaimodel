use crate::linalg::Vector;
use crate::nn::{LayerGradient, Network, NetworkError, NetworkGradients};
use std::error::Error;
use std::fmt;

/// Errors that can occur while configuring or running an optimizer.
#[derive(Debug, PartialEq)]
pub enum OptimizerError {
    /// Learning rate must be positive and finite.
    InvalidLearningRate { value: f64 },

    /// Momentum must be finite and in the range `0..1`.
    InvalidMomentum { value: f64 },

    /// Adam beta values must be finite and in the range `0..1`.
    InvalidBeta { value: f64 },

    /// Epsilon must be positive and finite.
    InvalidEpsilon { value: f64 },

    /// A network operation failed while applying gradients.
    Network(NetworkError),
}

impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerError::InvalidLearningRate { value } => {
                write!(f, "Learning rate must be positive and finite: {}", value)
            }
            OptimizerError::InvalidMomentum { value } => {
                write!(
                    f,
                    "Momentum must be finite and in the range 0..1: {}",
                    value
                )
            }
            OptimizerError::InvalidBeta { value } => {
                write!(f, "Beta must be finite and in the range 0..1: {}", value)
            }
            OptimizerError::InvalidEpsilon { value } => {
                write!(f, "Epsilon must be positive and finite: {}", value)
            }
            OptimizerError::Network(error) => write!(f, "Network optimizer error: {}", error),
        }
    }
}

impl Error for OptimizerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            OptimizerError::Network(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NetworkError> for OptimizerError {
    fn from(error: NetworkError) -> Self {
        OptimizerError::Network(error)
    }
}

/// Applies gradients to a network.
pub trait Optimizer {
    /// Applies one optimization step.
    fn step(
        &mut self,
        network: &mut Network,
        gradients: &NetworkGradients,
    ) -> Result<(), OptimizerError>;
}

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

fn validate_learning_rate(learning_rate: f64) -> Result<(), OptimizerError> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(OptimizerError::InvalidLearningRate {
            value: learning_rate,
        });
    }

    Ok(())
}

fn validate_momentum(momentum: f64) -> Result<(), OptimizerError> {
    if !momentum.is_finite() || !(0.0..1.0).contains(&momentum) {
        return Err(OptimizerError::InvalidMomentum { value: momentum });
    }

    Ok(())
}

fn validate_beta(beta: f64) -> Result<(), OptimizerError> {
    if !beta.is_finite() || !(0.0..1.0).contains(&beta) {
        return Err(OptimizerError::InvalidBeta { value: beta });
    }

    Ok(())
}

fn validate_epsilon(epsilon: f64) -> Result<(), OptimizerError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimizerError::InvalidEpsilon { value: epsilon });
    }

    Ok(())
}

fn ensure_state_shape(state: &mut Option<NetworkGradients>, gradients: &NetworkGradients) {
    if !state
        .as_ref()
        .is_some_and(|existing| same_gradient_shape(existing, gradients))
    {
        *state = Some(zeros_like(gradients));
    }
}

fn zeros_like(gradients: &NetworkGradients) -> NetworkGradients {
    NetworkGradients::new(
        gradients
            .layer_gradients()
            .iter()
            .map(|layer_gradient| {
                LayerGradient::new(
                    layer_gradient
                        .weight_gradients()
                        .iter()
                        .map(|weights| Vector::new(weights.iter().map(|_| 0.0).collect()))
                        .collect(),
                    Vector::new(
                        layer_gradient
                            .bias_gradients()
                            .iter()
                            .map(|_| 0.0)
                            .collect(),
                    ),
                )
            })
            .collect(),
    )
}

fn same_gradient_shape(left: &NetworkGradients, right: &NetworkGradients) -> bool {
    left.len() == right.len()
        && left
            .layer_gradients()
            .iter()
            .zip(right.layer_gradients().iter())
            .all(|(left_layer, right_layer)| {
                left_layer.bias_gradients().len() == right_layer.bias_gradients().len()
                    && left_layer.weight_gradients().len() == right_layer.weight_gradients().len()
                    && left_layer
                        .weight_gradients()
                        .iter()
                        .zip(right_layer.weight_gradients().iter())
                        .all(|(left_weights, right_weights)| {
                            left_weights.len() == right_weights.len()
                        })
            })
}

fn combine_gradients<F>(
    left: &NetworkGradients,
    right: &NetworkGradients,
    op: F,
) -> NetworkGradients
where
    F: Fn(f64, f64) -> f64 + Copy,
{
    NetworkGradients::new(
        left.layer_gradients()
            .iter()
            .zip(right.layer_gradients().iter())
            .map(|(left_layer, right_layer)| {
                LayerGradient::new(
                    left_layer
                        .weight_gradients()
                        .iter()
                        .zip(right_layer.weight_gradients().iter())
                        .map(|(left_weights, right_weights)| {
                            Vector::new(
                                left_weights
                                    .iter()
                                    .zip(right_weights.iter())
                                    .map(|(left_value, right_value)| op(*left_value, *right_value))
                                    .collect(),
                            )
                        })
                        .collect(),
                    Vector::new(
                        left_layer
                            .bias_gradients()
                            .iter()
                            .zip(right_layer.bias_gradients().iter())
                            .map(|(left_value, right_value)| op(*left_value, *right_value))
                            .collect(),
                    ),
                )
            })
            .collect(),
    )
}

fn adam_update(
    first_moment: &NetworkGradients,
    second_moment: &NetworkGradients,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    timestep: usize,
) -> NetworkGradients {
    let first_bias_correction = 1.0 - beta1.powi(timestep as i32);
    let second_bias_correction = 1.0 - beta2.powi(timestep as i32);

    combine_gradients(first_moment, second_moment, |first, second| {
        let corrected_first = first / first_bias_correction;
        let corrected_second = second / second_bias_correction;

        corrected_first / (corrected_second.sqrt() + epsilon)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::nn::{Layer, LayerGradient, NetworkLayer};

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-10, "{left} != {right}");
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
        assert_close(updated_prediction[0], 0.4);
    }

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
