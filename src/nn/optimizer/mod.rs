mod adam;
mod error;
mod momentum;
mod sgd;
mod traits;
mod utils;

pub use adam::Adam;
pub use error::OptimizerError;
pub use momentum::Momentum;
pub use sgd::Sgd;
pub use traits::Optimizer;

#[cfg(test)]
pub(crate) mod test_support {
    use crate::activations::Activation;
    use crate::linalg::Vector;
    use crate::nn::{Layer, Network, NetworkLayer};

    pub fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-10, "{left} != {right}");
    }

    pub fn dense_layer(weights: Vec<Vec<f64>>, biases: Vec<f64>) -> Layer {
        let weights = weights.into_iter().map(Vector::new).collect();

        Layer::dense(weights, biases).unwrap()
    }

    pub fn single_neuron_network(weight: f64, bias: f64) -> Network {
        Network::new(vec![NetworkLayer::new(
            dense_layer(vec![vec![weight]], vec![bias]),
            Activation::Linear,
        )])
        .unwrap()
    }
}
