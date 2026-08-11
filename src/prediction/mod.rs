//! Application-level DCG prediction APIs above raw neural-network vectors.

use crate::linalg::Vector;
use crate::nn::{Network, NetworkError};
use std::error::Error;
use std::fmt;

/// The interpretation applied to a single-output DCG network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionKind {
    /// Binary breaking-change classification.
    BreakingChange,
    /// Binary compatibility classification where `1` means incompatible.
    Compatibility,
    /// Regression over the normalized inclusive range `0.0..=1.0`.
    RiskScore,
}

/// A validated, high-level DCG prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelPrediction {
    /// A binary class prediction with its probability and thresholded label.
    Classification {
        /// Predicted probability of the positive class.
        probability: f64,
        /// Label selected with `probability >= threshold`.
        label: u8,
    },
    /// A normalized risk score.
    RiskScore {
        /// Predicted risk, where `0.0` is lowest and `1.0` highest.
        risk_score: f64,
    },
}

impl ModelPrediction {
    /// Returns the classification probability, if this is a classification prediction.
    pub fn probability(&self) -> Option<f64> {
        match self {
            Self::Classification { probability, .. } => Some(*probability),
            Self::RiskScore { .. } => None,
        }
    }

    /// Returns the thresholded binary label, if this is a classification prediction.
    pub fn label(&self) -> Option<u8> {
        match self {
            Self::Classification { label, .. } => Some(*label),
            Self::RiskScore { .. } => None,
        }
    }

    /// Returns the normalized risk score, if this is a risk prediction.
    pub fn risk_score(&self) -> Option<f64> {
        match self {
            Self::Classification { .. } => None,
            Self::RiskScore { risk_score } => Some(*risk_score),
        }
    }
}

/// Errors raised while validating a high-level prediction.
#[derive(Debug, PartialEq)]
pub enum PredictionError {
    /// Input feature count does not match the model input.
    FeatureDimensionMismatch { expected: usize, actual: usize },
    /// An input value was NaN or infinite.
    NonFiniteFeature { value: f64 },
    /// The wrapped network could not run.
    Network(NetworkError),
    /// The network output did not have exactly one value.
    OutputDimensionMismatch { expected: usize, actual: usize },
    /// The classification threshold was not finite and in `0.0..=1.0`.
    InvalidThreshold { value: f64 },
    /// A classification output was not a probability.
    InvalidProbability { value: f64 },
    /// A risk output was outside the documented normalized range.
    InvalidRiskScore { value: f64 },
}

impl fmt::Display for PredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FeatureDimensionMismatch { expected, actual } => write!(
                f,
                "Feature dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteFeature { value } => write!(f, "Non-finite input feature: {value}"),
            Self::Network(error) => write!(f, "Network prediction error: {error}"),
            Self::OutputDimensionMismatch { expected, actual } => write!(
                f,
                "Output dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidThreshold { value } => {
                write!(f, "Invalid classification threshold: {value}")
            }
            Self::InvalidProbability { value } => {
                write!(f, "Invalid classification probability: {value}")
            }
            Self::InvalidRiskScore { value } => write!(f, "Invalid normalized risk score: {value}"),
        }
    }
}

impl Error for PredictionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Network(error) => Some(error),
            _ => None,
        }
    }
}

impl From<NetworkError> for PredictionError {
    fn from(error: NetworkError) -> Self {
        Self::Network(error)
    }
}

/// Validates features and interprets a single-output DCG network result.
#[derive(Debug, Clone, Copy)]
pub struct Predictor<'a> {
    network: &'a Network,
    kind: PredictionKind,
    threshold: f64,
}

impl<'a> Predictor<'a> {
    /// Creates a predictor. Classification thresholds must be finite and within `0.0..=1.0`.
    pub fn new(
        network: &'a Network,
        kind: PredictionKind,
        threshold: f64,
    ) -> Result<Self, PredictionError> {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(PredictionError::InvalidThreshold { value: threshold });
        }
        Ok(Self {
            network,
            kind,
            threshold,
        })
    }

    /// Predicts one validated feature vector.
    pub fn predict(&self, features: &Vector) -> Result<ModelPrediction, PredictionError> {
        if features.len() != self.network.input_size() {
            return Err(PredictionError::FeatureDimensionMismatch {
                expected: self.network.input_size(),
                actual: features.len(),
            });
        }
        for value in features {
            if !value.is_finite() {
                return Err(PredictionError::NonFiniteFeature { value: *value });
            }
        }
        let output = self.network.forward(features)?;
        self.interpret_output(&output)
    }

    /// Predicts each vector in a batch using the same validation rules.
    pub fn predict_batch(
        &self,
        features: &[Vector],
    ) -> Result<Vec<ModelPrediction>, PredictionError> {
        features
            .iter()
            .map(|feature| self.predict(feature))
            .collect()
    }

    fn interpret_output(&self, output: &Vector) -> Result<ModelPrediction, PredictionError> {
        if output.len() != 1 {
            return Err(PredictionError::OutputDimensionMismatch {
                expected: 1,
                actual: output.len(),
            });
        }
        let value = output[0];
        if !value.is_finite() {
            return match self.kind {
                PredictionKind::RiskScore => Err(PredictionError::InvalidRiskScore { value }),
                _ => Err(PredictionError::InvalidProbability { value }),
            };
        }
        match self.kind {
            PredictionKind::BreakingChange | PredictionKind::Compatibility => {
                if !(0.0..=1.0).contains(&value) {
                    return Err(PredictionError::InvalidProbability { value });
                }
                Ok(ModelPrediction::Classification {
                    probability: value,
                    label: u8::from(value >= self.threshold),
                })
            }
            PredictionKind::RiskScore => {
                if !(0.0..=1.0).contains(&value) {
                    return Err(PredictionError::InvalidRiskScore { value });
                }
                Ok(ModelPrediction::RiskScore { risk_score: value })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::linalg::Vector;
    use crate::nn::{Layer, NetworkLayer};

    fn network(output_bias: f64) -> Network {
        Network::new(vec![NetworkLayer::new(
            Layer::dense(vec![Vector::new(vec![0.0])], vec![output_bias]).unwrap(),
            Activation::Linear,
        )])
        .unwrap()
    }

    #[test]
    fn classification_predicts_at_threshold_boundary() {
        let network = network(0.5);
        let prediction = Predictor::new(&network, PredictionKind::BreakingChange, 0.5)
            .unwrap()
            .predict(&Vector::new(vec![1.0]))
            .unwrap();
        assert_eq!(prediction.probability(), Some(0.5));
        assert_eq!(prediction.label(), Some(1));
    }

    #[test]
    fn validates_input_and_batch_predictions() {
        let network = network(0.2);
        let predictor = Predictor::new(&network, PredictionKind::Compatibility, 0.5).unwrap();
        assert!(matches!(
            predictor.predict(&Vector::new(vec![])),
            Err(PredictionError::FeatureDimensionMismatch { .. })
        ));
        assert!(
            matches!(predictor.predict(&Vector::new(vec![f64::NAN])), Err(PredictionError::NonFiniteFeature { value }) if value.is_nan())
        );
        assert_eq!(
            predictor
                .predict_batch(&[Vector::new(vec![1.0]), Vector::new(vec![2.0])])
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn rejects_invalid_model_output_and_threshold() {
        let invalid_network = network(1.2);
        let predictor =
            Predictor::new(&invalid_network, PredictionKind::BreakingChange, 0.5).unwrap();
        assert_eq!(
            predictor.predict(&Vector::new(vec![0.0])),
            Err(PredictionError::InvalidProbability { value: 1.2 })
        );
        assert!(
            matches!(Predictor::new(&invalid_network, PredictionKind::BreakingChange, f64::INFINITY), Err(PredictionError::InvalidThreshold { value }) if value.is_infinite())
        );
    }
}
