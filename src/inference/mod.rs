//! Inference-only runtime for validated persisted DCG models.

use crate::features::{
    ContractChange, ContractFeatureExtractor, DCG_FEATURE_VERSION, FeatureError, FeatureExtractor,
};
use crate::models::{ModelArtifact, ModelArtifactError};
use crate::prediction::ModelPrediction;
use std::error::Error;
use std::fmt;
use std::path::Path;

/// Structured advisory result returned by the inference-only runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionResult {
    /// Persisted trained-model identifier.
    pub model_version: String,
    /// Canonical schema used to extract the input.
    pub feature_version: &'static str,
    /// Validated task-specific model output.
    pub prediction: ModelPrediction,
    /// Classification threshold when applicable; risk scores do not use one.
    pub threshold: Option<f64>,
}

/// Errors raised by loading or using an inference-only runtime.
#[derive(Debug)]
pub enum InferenceError {
    /// The persisted model artifact was invalid or unavailable.
    Artifact(ModelArtifactError),
    /// Domain feature extraction or feature-schema validation failed.
    Feature(FeatureError),
    /// Applying stored preprocessing failed.
    Preprocessing(crate::dataset::DatasetError),
    /// The loaded model rejected the normalized input or output.
    Prediction(crate::models::ModelError),
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => write!(f, "Inference artifact error: {error}"),
            Self::Feature(error) => write!(f, "Inference feature error: {error}"),
            Self::Preprocessing(error) => write!(f, "Inference preprocessing error: {error}"),
            Self::Prediction(error) => write!(f, "Inference prediction error: {error}"),
        }
    }
}

impl Error for InferenceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Artifact(error) => Some(error),
            Self::Feature(error) => Some(error),
            Self::Preprocessing(error) => Some(error),
            Self::Prediction(error) => Some(error),
        }
    }
}

impl From<ModelArtifactError> for InferenceError {
    fn from(error: ModelArtifactError) -> Self {
        Self::Artifact(error)
    }
}
impl From<FeatureError> for InferenceError {
    fn from(error: FeatureError) -> Self {
        Self::Feature(error)
    }
}
impl From<crate::dataset::DatasetError> for InferenceError {
    fn from(error: crate::dataset::DatasetError) -> Self {
        Self::Preprocessing(error)
    }
}
impl From<crate::models::ModelError> for InferenceError {
    fn from(error: crate::models::ModelError) -> Self {
        Self::Prediction(error)
    }
}

/// Loads a persisted artifact and only performs validation, scaling, and prediction.
#[derive(Debug, Clone)]
pub struct InferenceRuntime {
    artifact: ModelArtifact,
}

impl InferenceRuntime {
    /// Loads a validated model artifact without training or fitting preprocessing.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, InferenceError> {
        Ok(Self {
            artifact: ModelArtifact::load(path)?,
        })
    }

    /// Creates an inference runtime from an already validated artifact.
    pub fn new(artifact: ModelArtifact) -> Self {
        Self { artifact }
    }

    /// Extracts canonical features, applies the stored scaler, and predicts.
    ///
    /// This method never mutates model weights or scaler state and never sees
    /// training data. The result is advisory; deterministic DCG policy remains
    /// authoritative for enforcement.
    pub fn predict_contract(
        &self,
        contract: &ContractChange,
    ) -> Result<PredictionResult, InferenceError> {
        let features = ContractFeatureExtractor::new().extract(contract)?;
        ContractFeatureExtractor::validate_feature_vector(&features)?;
        let normalized = self.artifact.scaler().transform_vector(&features)?;
        let prediction = self.artifact.model().predict(&normalized)?;
        let threshold = match prediction {
            ModelPrediction::Classification { .. } => Some(self.artifact.model().threshold()),
            ModelPrediction::RiskScore { .. } => None,
        };
        Ok(PredictionResult {
            model_version: self.artifact.model_version().to_owned(),
            feature_version: DCG_FEATURE_VERSION,
            prediction,
            threshold,
        })
    }

    /// Returns the model artifact backing this immutable inference path.
    pub fn artifact(&self) -> &ModelArtifact {
        &self.artifact
    }
}
