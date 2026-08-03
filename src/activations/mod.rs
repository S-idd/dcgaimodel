mod activation;
mod relu;
mod sigmoid;
mod softmax;
mod tanh;

pub use activation::Activation;
pub use relu::{relu, relu_vector};
pub use sigmoid::{sigmoid, sigmoid_vector};
pub use softmax::softmax;
pub use tanh::{tanh, tanh_vector};
