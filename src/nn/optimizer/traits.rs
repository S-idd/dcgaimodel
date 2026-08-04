use crate::nn::{Network, NetworkGradients, OptimizerError};

/// Applies gradients to a network, mutating its parameters in place.
///
/// Implementors define how raw gradients are transformed into a parameter
/// update — e.g. plain SGD, momentum, or an adaptive method like Adam.
pub trait Optimizer {
    /// Applies one optimization step to `network` using `gradients`.
    fn step(
        &mut self,
        network: &mut Network,
        gradients: &NetworkGradients,
    ) -> Result<(), OptimizerError>;
}
