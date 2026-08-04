use super::error::OptimizerError;
use crate::linalg::Vector;
use crate::nn::{LayerGradient, NetworkGradients};

/// Validates that a learning rate is positive and finite.
pub(super) fn validate_learning_rate(learning_rate: f64) -> Result<(), OptimizerError> {
    if !learning_rate.is_finite() || learning_rate <= 0.0 {
        return Err(OptimizerError::InvalidLearningRate {
            value: learning_rate,
        });
    }

    Ok(())
}

/// Validates that a momentum coefficient is finite and in `0..1`.
pub(super) fn validate_momentum(momentum: f64) -> Result<(), OptimizerError> {
    if !momentum.is_finite() || !(0.0..1.0).contains(&momentum) {
        return Err(OptimizerError::InvalidMomentum { value: momentum });
    }

    Ok(())
}

/// Validates that an Adam beta coefficient is finite and in `0..1`.
pub(super) fn validate_beta(beta: f64) -> Result<(), OptimizerError> {
    if !beta.is_finite() || !(0.0..1.0).contains(&beta) {
        return Err(OptimizerError::InvalidBeta { value: beta });
    }

    Ok(())
}

/// Validates that an epsilon value is positive and finite.
pub(super) fn validate_epsilon(epsilon: f64) -> Result<(), OptimizerError> {
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(OptimizerError::InvalidEpsilon { value: epsilon });
    }

    Ok(())
}

/// Resets `state` to a zeroed shape matching `gradients` if it is absent or
/// no longer matches the gradient shape.
pub(super) fn ensure_state_shape(
    state: &mut Option<NetworkGradients>,
    gradients: &NetworkGradients,
) {
    if !state
        .as_ref()
        .is_some_and(|existing| same_gradient_shape(existing, gradients))
    {
        *state = Some(zeros_like(gradients));
    }
}

/// Builds a `NetworkGradients` with the same shape as `gradients`, but with
/// every value set to zero.
pub(super) fn zeros_like(gradients: &NetworkGradients) -> NetworkGradients {
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

/// Returns true when two gradient sets have the same layer/weight/bias
/// shapes.
pub(super) fn same_gradient_shape(left: &NetworkGradients, right: &NetworkGradients) -> bool {
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

/// Combines two gradient sets element-wise using `op`.
pub(super) fn combine_gradients<F>(
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

/// Computes the bias-corrected Adam update from first/second moment
/// estimates.
pub(super) fn adam_update(
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
