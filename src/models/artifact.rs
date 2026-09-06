//! Explicit, versioned persistence for inference-ready DCG models.

use super::{DcgModel, ModelError, ThreeWayCompatibilityModel};
use crate::activations::Activation;
use crate::features::{DCG_FEATURE_VERSION, feature_count_for_version};
use crate::linalg::Vector;
use crate::nn::{Layer, Network, NetworkLayer};
use crate::prediction::PredictionKind;
use crate::preprocessing::StandardScaler;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

/// Stable format identifier for persisted `dcgaimodel` artifacts.
pub const ARTIFACT_FORMAT_VERSION: &str = "dcgaimodel-artifact-v1";

/// Stable format identifier for persisted three-way Softmax DCG artifacts.
///
/// This is intentionally distinct from `ARTIFACT_FORMAT_VERSION`: existing
/// binary artifacts remain readable without changing their schema or meaning.
pub const THREE_WAY_ARTIFACT_FORMAT_VERSION: &str = "dcgaimodel-three-way-artifact-v1";

/// Canonical, non-configurable Softmax output ordering for compatibility
/// inference. Artifact files retain this explicitly so consumers cannot
/// accidentally reinterpret output positions.
pub const THREE_WAY_CLASS_ORDER: [&str; 3] = ["safe", "warning", "breaking"];

/// Immutable identities of the exact inputs used to produce a serious
/// oracle-labelled training run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrainingInputProvenance {
    /// SHA-256 of the saved prepared-dataset artifact.
    pub dataset_sha256: String,
    /// SHA-256 of the executable oracle JAR that labelled the dataset.
    pub oracle_jar_sha256: String,
    /// SHA-256 of the policy-pack file used by that executable oracle.
    pub policy_packs_sha256: String,
}

impl TrainingInputProvenance {
    /// Validates complete, canonical SHA-256 identities.
    pub fn validate(self) -> Result<Self, ModelArtifactError> {
        for (field, value) in [
            ("dataset_sha256", &self.dataset_sha256),
            ("oracle_jar_sha256", &self.oracle_jar_sha256),
            ("policy_packs_sha256", &self.policy_packs_sha256),
        ] {
            if value.len() != 64
                || !value
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
            {
                return Err(ModelArtifactError::InvalidState {
                    reason: format!("{field} must be a 64-character SHA-256 hex digest"),
                });
            }
        }
        Ok(self)
    }
}

/// Training context stored alongside a model for reproducibility and auditability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingMetadata {
    /// Optimizer name supplied by the caller (for example, `sgd`).
    pub optimizer: String,
    /// Completed training epochs.
    pub epochs: usize,
    /// Training batch size.
    pub batch_size: usize,
    /// Optional learning-rate representation when the caller records it.
    pub learning_rate: Option<String>,
    /// Dataset split/training seed. Current model construction and batching are deterministic.
    pub seed: u64,
    /// Number of samples used for fitting model parameters.
    pub training_samples: usize,
    /// Number of samples used for validation/model selection.
    pub validation_samples: usize,
    /// Number of samples reserved for final evaluation.
    pub test_samples: usize,
    /// Portable dataset or fixture-set identifier.
    pub dataset_id: String,
}

impl TrainingMetadata {
    /// Validates metadata constructed from explicitly named fields.
    pub fn validate(self) -> Result<Self, ModelArtifactError> {
        if self.optimizer.trim().is_empty()
            || self.dataset_id.trim().is_empty()
            || self.epochs == 0
            || self.batch_size == 0
        {
            return Err(ModelArtifactError::InvalidMetadata);
        }
        Ok(self)
    }
}

/// Errors produced while saving, loading, or validating a model artifact.
#[derive(Debug)]
pub enum ModelArtifactError {
    /// Disk IO failed.
    Io { message: String },
    /// Artifact JSON was malformed or missing required data.
    Serialization { message: String },
    /// The artifact format is not supported by this runtime.
    UnsupportedArtifactVersion { actual: String },
    /// The persisted schema does not match the runtime's canonical schema.
    FeatureVersionMismatch {
        expected: &'static str,
        actual: String,
    },
    /// A required model identifier was invalid.
    InvalidModelVersion,
    /// Model kind was not recognized.
    InvalidModelKind { actual: String },
    /// Activation was not recognized.
    InvalidActivation { actual: String },
    /// A dimension, numeric parameter, or threshold was invalid.
    InvalidState { reason: String },
    /// Persisted metadata was incomplete or invalid.
    InvalidMetadata,
    /// Reconstructing the model through existing APIs failed.
    Model(ModelError),
    /// Reconstructing scaler state failed.
    Dataset(crate::dataset::DatasetError),
}

impl fmt::Display for ModelArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { message } => write!(f, "Model artifact IO error: {message}"),
            Self::Serialization { message } => write!(f, "Invalid model artifact: {message}"),
            Self::UnsupportedArtifactVersion { actual } => {
                write!(f, "Unsupported model artifact version: {actual}")
            }
            Self::FeatureVersionMismatch { expected, actual } => write!(
                f,
                "Feature version mismatch: runtime expects {expected}, artifact has {actual}"
            ),
            Self::InvalidModelVersion => write!(f, "Model version must not be empty."),
            Self::InvalidModelKind { actual } => write!(f, "Unknown model kind: {actual}"),
            Self::InvalidActivation { actual } => write!(f, "Unknown activation: {actual}"),
            Self::InvalidState { reason } => write!(f, "Invalid model artifact state: {reason}"),
            Self::InvalidMetadata => write!(f, "Invalid training metadata."),
            Self::Model(error) => write!(f, "Persisted model is invalid: {error}"),
            Self::Dataset(error) => write!(f, "Persisted scaler is invalid: {error}"),
        }
    }
}

impl Error for ModelArtifactError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            Self::Dataset(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ModelError> for ModelArtifactError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}
impl From<crate::dataset::DatasetError> for ModelArtifactError {
    fn from(error: crate::dataset::DatasetError) -> Self {
        Self::Dataset(error)
    }
}

/// A validated in-memory artifact, ready for independent inference.
#[derive(Debug, Clone)]
pub struct ModelArtifact {
    model_version: String,
    feature_version: String,
    model: DcgModel,
    scaler: StandardScaler,
    training_metadata: TrainingMetadata,
}

/// A validated, inference-ready SAFE/WARNING/BREAKING Softmax model.
///
/// The artifact contains the fitted training-only scaler, immutable input
/// identities, and the canonical output ordering needed for portable
/// three-class inference.
#[derive(Debug, Clone)]
pub struct ThreeWayModelArtifact {
    model_version: String,
    feature_version: String,
    model: ThreeWayCompatibilityModel,
    scaler: StandardScaler,
    training_metadata: TrainingMetadata,
    input_provenance: TrainingInputProvenance,
}

impl ModelArtifact {
    /// Creates an artifact from a trained model and the scaler fitted on its training partition.
    pub fn new(
        model_version: impl Into<String>,
        model: DcgModel,
        scaler: StandardScaler,
        training_metadata: TrainingMetadata,
    ) -> Result<Self, ModelArtifactError> {
        Self::new_with_feature_version(
            model_version,
            DCG_FEATURE_VERSION,
            model,
            scaler,
            training_metadata,
        )
    }

    /// Creates an artifact explicitly bound to one validated feature schema.
    pub fn new_with_feature_version(
        model_version: impl Into<String>,
        feature_version: impl Into<String>,
        model: DcgModel,
        scaler: StandardScaler,
        training_metadata: TrainingMetadata,
    ) -> Result<Self, ModelArtifactError> {
        let model_version = model_version.into();
        let feature_version = feature_version.into();
        if model_version.trim().is_empty() {
            return Err(ModelArtifactError::InvalidModelVersion);
        }
        validate_model_scaler(&model, &scaler, &feature_version)?;
        Ok(Self {
            model_version,
            feature_version,
            model,
            scaler,
            training_metadata,
        })
    }

    /// Returns the independently versioned trained model identifier.
    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    /// Returns the feature schema required by this artifact.
    pub fn feature_version(&self) -> &str {
        &self.feature_version
    }

    /// Returns the inference-ready model.
    pub fn model(&self) -> &DcgModel {
        &self.model
    }

    /// Returns the fitted training scaler.
    pub fn scaler(&self) -> &StandardScaler {
        &self.scaler
    }

    /// Returns portable training metadata.
    pub fn training_metadata(&self) -> &TrainingMetadata {
        &self.training_metadata
    }

    /// Writes a complete JSON artifact via a temporary sibling file and rename.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), ModelArtifactError> {
        validate_model_scaler(&self.model, &self.scaler, &self.feature_version)?;
        let file = ArtifactFile::from_artifact(self);
        let contents = serde_json::to_vec_pretty(&file).map_err(|error| {
            ModelArtifactError::Serialization {
                message: error.to_string(),
            }
        })?;
        let path = path.as_ref();
        let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
        fs::write(&temporary, contents).map_err(|error| ModelArtifactError::Io {
            message: error.to_string(),
        })?;
        fs::rename(&temporary, path).map_err(|error| ModelArtifactError::Io {
            message: error.to_string(),
        })?;
        Ok(())
    }

    /// Loads and validates a complete artifact without training or refitting preprocessing.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ModelArtifactError> {
        let contents = fs::read(path).map_err(|error| ModelArtifactError::Io {
            message: error.to_string(),
        })?;
        let file = serde_json::from_slice::<ArtifactFile>(&contents).map_err(|error| {
            ModelArtifactError::Serialization {
                message: error.to_string(),
            }
        })?;
        file.into_artifact()
    }
}

impl ThreeWayModelArtifact {
    /// Creates a three-way artifact explicitly bound to one feature schema.
    pub fn new_with_feature_version(
        model_version: impl Into<String>,
        feature_version: impl Into<String>,
        model: ThreeWayCompatibilityModel,
        scaler: StandardScaler,
        training_metadata: TrainingMetadata,
        input_provenance: TrainingInputProvenance,
    ) -> Result<Self, ModelArtifactError> {
        let model_version = model_version.into();
        let feature_version = feature_version.into();
        if model_version.trim().is_empty() {
            return Err(ModelArtifactError::InvalidModelVersion);
        }
        validate_three_way_model_scaler(&model, &scaler, &feature_version)?;
        Ok(Self {
            model_version,
            feature_version,
            model,
            scaler,
            training_metadata: training_metadata.validate()?,
            input_provenance: input_provenance.validate()?,
        })
    }

    /// Returns the independently versioned model identifier.
    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    /// Returns the feature schema required by this artifact.
    pub fn feature_version(&self) -> &str {
        &self.feature_version
    }

    /// Returns the inference-ready three-class model.
    pub fn model(&self) -> &ThreeWayCompatibilityModel {
        &self.model
    }

    /// Returns the training-only fitted scaler.
    pub fn scaler(&self) -> &StandardScaler {
        &self.scaler
    }

    /// Returns portable optimizer and split metadata.
    pub fn training_metadata(&self) -> &TrainingMetadata {
        &self.training_metadata
    }

    /// Returns the immutable dataset/oracle/policy identities.
    pub fn input_provenance(&self) -> &TrainingInputProvenance {
        &self.input_provenance
    }

    /// Writes a complete JSON artifact through a temporary sibling file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), ModelArtifactError> {
        validate_three_way_model_scaler(&self.model, &self.scaler, &self.feature_version)?;
        let file = ThreeWayArtifactFile::from_artifact(self);
        let contents = serde_json::to_vec_pretty(&file).map_err(|error| {
            ModelArtifactError::Serialization {
                message: error.to_string(),
            }
        })?;
        let path = path.as_ref();
        let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
        fs::write(&temporary, contents).map_err(|error| ModelArtifactError::Io {
            message: error.to_string(),
        })?;
        fs::rename(&temporary, path).map_err(|error| ModelArtifactError::Io {
            message: error.to_string(),
        })?;
        Ok(())
    }

    /// Loads and validates a three-way artifact without refitting preprocessing.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ModelArtifactError> {
        let contents = fs::read(path).map_err(|error| ModelArtifactError::Io {
            message: error.to_string(),
        })?;
        Self::from_slice(&contents)
    }

    /// Validates a three-way artifact from already authenticated JSON bytes.
    pub fn from_slice(contents: &[u8]) -> Result<Self, ModelArtifactError> {
        let file = serde_json::from_slice::<ThreeWayArtifactFile>(contents).map_err(|error| {
            ModelArtifactError::Serialization {
                message: error.to_string(),
            }
        })?;
        file.into_artifact()
    }
}

fn validate_model_scaler(
    model: &DcgModel,
    scaler: &StandardScaler,
    feature_version: &str,
) -> Result<(), ModelArtifactError> {
    let Some(feature_count) = feature_count_for_version(feature_version) else {
        return Err(ModelArtifactError::FeatureVersionMismatch {
            expected: DCG_FEATURE_VERSION,
            actual: feature_version.to_owned(),
        });
    };
    if model.network().input_size() != feature_count {
        return Err(ModelArtifactError::InvalidState {
            reason: format!(
                "model expects {} inputs, but {feature_count} are required",
                model.network().input_size()
            ),
        });
    }
    if scaler.means().len() != feature_count || scaler.standard_deviations().len() != feature_count
    {
        return Err(ModelArtifactError::InvalidState {
            reason: "scaler dimensions do not match feature schema".to_owned(),
        });
    }
    Ok(())
}

fn validate_three_way_model_scaler(
    model: &ThreeWayCompatibilityModel,
    scaler: &StandardScaler,
    feature_version: &str,
) -> Result<(), ModelArtifactError> {
    let Some(feature_count) = feature_count_for_version(feature_version) else {
        return Err(ModelArtifactError::FeatureVersionMismatch {
            expected: DCG_FEATURE_VERSION,
            actual: feature_version.to_owned(),
        });
    };
    if model.network().input_size() != feature_count {
        return Err(ModelArtifactError::InvalidState {
            reason: format!(
                "model expects {} inputs, but {feature_count} are required",
                model.network().input_size()
            ),
        });
    }
    if scaler.means().len() != feature_count || scaler.standard_deviations().len() != feature_count
    {
        return Err(ModelArtifactError::InvalidState {
            reason: "scaler dimensions do not match feature schema".to_owned(),
        });
    }
    if model.network().output_size() != 3
        || model
            .network()
            .layers()
            .last()
            .map(NetworkLayer::activation)
            != Some(Activation::Softmax)
    {
        return Err(ModelArtifactError::InvalidState {
            reason: "three-way model must have a three-output Softmax head".to_owned(),
        });
    }
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
struct ArtifactFile {
    artifact_version: String,
    model_version: String,
    feature_version: String,
    model_kind: String,
    threshold: f64,
    network: Vec<LayerFile>,
    scaler: ScalerFile,
    training_metadata: MetadataFile,
}

#[derive(Debug, Serialize, Deserialize)]
struct ThreeWayArtifactFile {
    artifact_version: String,
    model_version: String,
    feature_version: String,
    class_order: Vec<String>,
    network: Vec<LayerFile>,
    scaler: ScalerFile,
    training_metadata: MetadataFile,
    input_provenance: InputProvenanceFile,
}

#[derive(Debug, Serialize, Deserialize)]
struct LayerFile {
    activation: String,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ScalerFile {
    means: Vec<f64>,
    standard_deviations: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct MetadataFile {
    optimizer: String,
    epochs: usize,
    batch_size: usize,
    learning_rate: Option<String>,
    seed: u64,
    training_samples: usize,
    validation_samples: usize,
    test_samples: usize,
    dataset_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct InputProvenanceFile {
    dataset_sha256: String,
    oracle_jar_sha256: String,
    policy_packs_sha256: String,
}

impl ArtifactFile {
    fn from_artifact(artifact: &ModelArtifact) -> Self {
        Self {
            artifact_version: ARTIFACT_FORMAT_VERSION.to_owned(),
            model_version: artifact.model_version.clone(),
            feature_version: artifact.feature_version.clone(),
            model_kind: kind_name(artifact.model.kind()).to_owned(),
            threshold: artifact.model.threshold(),
            network: artifact
                .model
                .network()
                .layers()
                .iter()
                .map(|layer| LayerFile {
                    activation: activation_name(layer.activation()).to_owned(),
                    weights: layer
                        .layer()
                        .neurons()
                        .iter()
                        .map(|neuron| neuron.weights().iter().copied().collect())
                        .collect(),
                    biases: layer
                        .layer()
                        .neurons()
                        .iter()
                        .map(|neuron| neuron.bias())
                        .collect(),
                })
                .collect(),
            scaler: ScalerFile {
                means: artifact.scaler.means().iter().copied().collect(),
                standard_deviations: artifact
                    .scaler
                    .standard_deviations()
                    .iter()
                    .copied()
                    .collect(),
            },
            training_metadata: MetadataFile::from(&artifact.training_metadata),
        }
    }

    fn into_artifact(self) -> Result<ModelArtifact, ModelArtifactError> {
        if self.artifact_version != ARTIFACT_FORMAT_VERSION {
            return Err(ModelArtifactError::UnsupportedArtifactVersion {
                actual: self.artifact_version,
            });
        }
        if feature_count_for_version(&self.feature_version).is_none() {
            return Err(ModelArtifactError::FeatureVersionMismatch {
                expected: DCG_FEATURE_VERSION,
                actual: self.feature_version,
            });
        }
        if self.model_version.trim().is_empty() {
            return Err(ModelArtifactError::InvalidModelVersion);
        }
        let kind = parse_kind(&self.model_kind)?;
        if !self.threshold.is_finite() || !(0.0..=1.0).contains(&self.threshold) {
            return Err(ModelArtifactError::InvalidState {
                reason: "threshold must be finite and within 0.0..=1.0".to_owned(),
            });
        }
        let layers = self
            .network
            .into_iter()
            .enumerate()
            .map(|(index, layer)| layer.into_network_layer(index))
            .collect::<Result<Vec<_>, _>>()?;
        let network = Network::new(layers).map_err(|error| ModelArtifactError::InvalidState {
            reason: error.to_string(),
        })?;
        let model = DcgModel::new(kind, network, self.threshold)?;
        let scaler = StandardScaler::from_statistics(
            Vector::new(self.scaler.means),
            Vector::new(self.scaler.standard_deviations),
        )?;
        let training_metadata = TrainingMetadata::try_from(self.training_metadata)?;
        ModelArtifact::new_with_feature_version(
            self.model_version,
            self.feature_version,
            model,
            scaler,
            training_metadata,
        )
    }
}

impl ThreeWayArtifactFile {
    fn from_artifact(artifact: &ThreeWayModelArtifact) -> Self {
        Self {
            artifact_version: THREE_WAY_ARTIFACT_FORMAT_VERSION.to_owned(),
            model_version: artifact.model_version.clone(),
            feature_version: artifact.feature_version.clone(),
            class_order: THREE_WAY_CLASS_ORDER
                .iter()
                .map(|label| (*label).to_owned())
                .collect(),
            network: serialize_network(artifact.model.network()),
            scaler: ScalerFile {
                means: artifact.scaler.means().iter().copied().collect(),
                standard_deviations: artifact
                    .scaler
                    .standard_deviations()
                    .iter()
                    .copied()
                    .collect(),
            },
            training_metadata: MetadataFile::from(&artifact.training_metadata),
            input_provenance: InputProvenanceFile::from(&artifact.input_provenance),
        }
    }

    fn into_artifact(self) -> Result<ThreeWayModelArtifact, ModelArtifactError> {
        if self.artifact_version != THREE_WAY_ARTIFACT_FORMAT_VERSION {
            return Err(ModelArtifactError::UnsupportedArtifactVersion {
                actual: self.artifact_version,
            });
        }
        if self.class_order
            != THREE_WAY_CLASS_ORDER
                .iter()
                .map(|label| (*label).to_owned())
                .collect::<Vec<_>>()
        {
            return Err(ModelArtifactError::InvalidState {
                reason: "three-way class order must be [safe, warning, breaking]".to_owned(),
            });
        }
        let layers = self
            .network
            .into_iter()
            .enumerate()
            .map(|(index, layer)| layer.into_network_layer(index))
            .collect::<Result<Vec<_>, _>>()?;
        let network = Network::new(layers).map_err(|error| ModelArtifactError::InvalidState {
            reason: error.to_string(),
        })?;
        let model = ThreeWayCompatibilityModel::new(network)?;
        let scaler = StandardScaler::from_statistics(
            Vector::new(self.scaler.means),
            Vector::new(self.scaler.standard_deviations),
        )?;
        ThreeWayModelArtifact::new_with_feature_version(
            self.model_version,
            self.feature_version,
            model,
            scaler,
            TrainingMetadata::try_from(self.training_metadata)?,
            TrainingInputProvenance::try_from(self.input_provenance)?,
        )
    }
}

fn serialize_network(network: &Network) -> Vec<LayerFile> {
    network
        .layers()
        .iter()
        .map(|layer| LayerFile {
            activation: activation_name(layer.activation()).to_owned(),
            weights: layer
                .layer()
                .neurons()
                .iter()
                .map(|neuron| neuron.weights().iter().copied().collect())
                .collect(),
            biases: layer
                .layer()
                .neurons()
                .iter()
                .map(|neuron| neuron.bias())
                .collect(),
        })
        .collect()
}

impl LayerFile {
    fn into_network_layer(self, index: usize) -> Result<NetworkLayer, ModelArtifactError> {
        if self.weights.is_empty() || self.weights.len() != self.biases.len() {
            return Err(ModelArtifactError::InvalidState {
                reason: format!("layer {index} has incompatible weight and bias counts"),
            });
        }
        let input_size = self.weights[0].len();
        if input_size == 0 {
            return Err(ModelArtifactError::InvalidState {
                reason: format!("layer {index} has no input weights"),
            });
        }
        for (neuron, weights) in self.weights.iter().enumerate() {
            if weights.len() != input_size {
                return Err(ModelArtifactError::InvalidState {
                    reason: format!("layer {index}, neuron {neuron} has inconsistent weight count"),
                });
            }
            for value in weights {
                if !value.is_finite() {
                    return Err(ModelArtifactError::InvalidState {
                        reason: format!("layer {index} contains non-finite weight"),
                    });
                }
            }
        }
        if self.biases.iter().any(|bias| !bias.is_finite()) {
            return Err(ModelArtifactError::InvalidState {
                reason: format!("layer {index} contains non-finite bias"),
            });
        }
        let activation = parse_activation(&self.activation)?;
        let layer = Layer::dense(
            self.weights.into_iter().map(Vector::new).collect(),
            self.biases,
        )
        .map_err(|error| ModelArtifactError::InvalidState {
            reason: error.to_string(),
        })?;
        Ok(NetworkLayer::new(layer, activation))
    }
}

impl From<&TrainingMetadata> for MetadataFile {
    fn from(value: &TrainingMetadata) -> Self {
        Self {
            optimizer: value.optimizer.clone(),
            epochs: value.epochs,
            batch_size: value.batch_size,
            learning_rate: value.learning_rate.clone(),
            seed: value.seed,
            training_samples: value.training_samples,
            validation_samples: value.validation_samples,
            test_samples: value.test_samples,
            dataset_id: value.dataset_id.clone(),
        }
    }
}

impl TryFrom<MetadataFile> for TrainingMetadata {
    type Error = ModelArtifactError;

    fn try_from(value: MetadataFile) -> Result<Self, Self::Error> {
        Self {
            optimizer: value.optimizer,
            epochs: value.epochs,
            batch_size: value.batch_size,
            learning_rate: value.learning_rate,
            seed: value.seed,
            training_samples: value.training_samples,
            validation_samples: value.validation_samples,
            test_samples: value.test_samples,
            dataset_id: value.dataset_id,
        }
        .validate()
    }
}

impl From<&TrainingInputProvenance> for InputProvenanceFile {
    fn from(value: &TrainingInputProvenance) -> Self {
        Self {
            dataset_sha256: value.dataset_sha256.clone(),
            oracle_jar_sha256: value.oracle_jar_sha256.clone(),
            policy_packs_sha256: value.policy_packs_sha256.clone(),
        }
    }
}

impl TryFrom<InputProvenanceFile> for TrainingInputProvenance {
    type Error = ModelArtifactError;

    fn try_from(value: InputProvenanceFile) -> Result<Self, Self::Error> {
        Self {
            dataset_sha256: value.dataset_sha256,
            oracle_jar_sha256: value.oracle_jar_sha256,
            policy_packs_sha256: value.policy_packs_sha256,
        }
        .validate()
    }
}

fn kind_name(kind: PredictionKind) -> &'static str {
    match kind {
        PredictionKind::BreakingChange => "breaking_change",
        PredictionKind::Compatibility => "compatibility",
        PredictionKind::RiskScore => "risk_score",
    }
}

fn parse_kind(value: &str) -> Result<PredictionKind, ModelArtifactError> {
    match value {
        "breaking_change" => Ok(PredictionKind::BreakingChange),
        "compatibility" => Ok(PredictionKind::Compatibility),
        "risk_score" => Ok(PredictionKind::RiskScore),
        _ => Err(ModelArtifactError::InvalidModelKind {
            actual: value.to_owned(),
        }),
    }
}

fn activation_name(activation: Activation) -> &'static str {
    match activation {
        Activation::Linear => "linear",
        Activation::Relu => "relu",
        Activation::Sigmoid => "sigmoid",
        Activation::Tanh => "tanh",
        Activation::Softmax => "softmax",
    }
}

fn parse_activation(value: &str) -> Result<Activation, ModelArtifactError> {
    match value {
        "linear" => Ok(Activation::Linear),
        "relu" => Ok(Activation::Relu),
        "sigmoid" => Ok(Activation::Sigmoid),
        "tanh" => Ok(Activation::Tanh),
        "softmax" => Ok(Activation::Softmax),
        _ => Err(ModelArtifactError::InvalidActivation {
            actual: value.to_owned(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::features::DCG_FEATURE_COUNT;
    use crate::nn::Sgd;
    use crate::training::TrainingConfig;

    fn artifact() -> ModelArtifact {
        let dataset = Dataset::new(
            vec![
                Vector::new(vec![0.0; DCG_FEATURE_COUNT]),
                Vector::new(vec![1.0; DCG_FEATURE_COUNT]),
            ],
            vec![Vector::new(vec![0.0]), Vector::new(vec![1.0])],
        )
        .unwrap();
        let scaler = StandardScaler::fit(&dataset).unwrap();
        let mut model = DcgModel::with_default_network(
            PredictionKind::BreakingChange,
            super::super::ModelConfig::default(),
        )
        .unwrap();
        model
            .train(
                &scaler.transform_dataset(&dataset).unwrap(),
                Sgd::new(0.05).unwrap(),
                TrainingConfig::new(2, 1).unwrap(),
            )
            .unwrap();
        ModelArtifact::new(
            "dcg-breaking-model-v1",
            model,
            scaler,
            TrainingMetadata {
                optimizer: "sgd".to_owned(),
                epochs: 2,
                batch_size: 1,
                learning_rate: None,
                seed: 17,
                training_samples: 2,
                validation_samples: 0,
                test_samples: 0,
                dataset_id: "artifact-test-v1".to_owned(),
            }
            .validate()
            .unwrap(),
        )
        .unwrap()
    }

    fn three_way_artifact() -> ThreeWayModelArtifact {
        let dataset = Dataset::new(
            vec![
                Vector::new(vec![0.0; DCG_FEATURE_COUNT]),
                Vector::new(vec![1.0; DCG_FEATURE_COUNT]),
                Vector::new(vec![2.0; DCG_FEATURE_COUNT]),
            ],
            vec![
                Vector::new(vec![1.0, 0.0, 0.0]),
                Vector::new(vec![0.0, 1.0, 0.0]),
                Vector::new(vec![0.0, 0.0, 1.0]),
            ],
        )
        .unwrap();
        let scaler = StandardScaler::fit(&dataset).unwrap();
        let mut model =
            ThreeWayCompatibilityModel::with_default_network(super::super::ModelConfig::default())
                .unwrap();
        model
            .train_with_validation(
                &scaler.transform_dataset(&dataset).unwrap(),
                None,
                Sgd::new(0.05).unwrap(),
                TrainingConfig::new(2, 1).unwrap(),
            )
            .unwrap();
        ThreeWayModelArtifact::new_with_feature_version(
            "dcg-three-way-model-v1",
            DCG_FEATURE_VERSION,
            model,
            scaler,
            TrainingMetadata {
                optimizer: "sgd".to_owned(),
                epochs: 2,
                batch_size: 1,
                learning_rate: Some("0.05".to_owned()),
                seed: 17,
                training_samples: 3,
                validation_samples: 0,
                test_samples: 0,
                dataset_id: "three-way-artifact-test-v1".to_owned(),
            }
            .validate()
            .unwrap(),
            TrainingInputProvenance {
                dataset_sha256: "a".repeat(64),
                oracle_jar_sha256: "b".repeat(64),
                policy_packs_sha256: "c".repeat(64),
            },
        )
        .unwrap()
    }

    #[test]
    fn persists_and_restores_network_and_scaler_state() {
        let artifact = artifact();
        let path =
            std::env::temp_dir().join(format!("dcgaimodel-artifact-{}.json", std::process::id()));
        artifact.save(&path).unwrap();
        let restored = ModelArtifact::load(&path).unwrap();
        let input = Vector::new(vec![0.5; DCG_FEATURE_COUNT]);
        let original = artifact
            .model()
            .predict(&artifact.scaler().transform_vector(&input).unwrap())
            .unwrap();
        let loaded = restored
            .model()
            .predict(&restored.scaler().transform_vector(&input).unwrap())
            .unwrap();
        assert_eq!(original, loaded);
        assert_eq!(restored.training_metadata().seed, 17);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn persists_and_restores_three_way_model_and_input_provenance() {
        let artifact = three_way_artifact();
        let path = std::env::temp_dir().join(format!(
            "dcgaimodel-three-way-artifact-{}.json",
            std::process::id()
        ));
        artifact.save(&path).unwrap();
        let restored = ThreeWayModelArtifact::load(&path).unwrap();
        let input = Vector::new(vec![0.5; DCG_FEATURE_COUNT]);
        let original = artifact
            .model()
            .predict(&artifact.scaler().transform_vector(&input).unwrap())
            .unwrap();
        let loaded = restored
            .model()
            .predict(&restored.scaler().transform_vector(&input).unwrap())
            .unwrap();
        assert_eq!(original, loaded);
        assert_eq!(restored.input_provenance().dataset_sha256, "a".repeat(64));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_three_way_artifact_with_reordered_classes() {
        let path = std::env::temp_dir().join(format!(
            "dcgaimodel-three-way-artifact-invalid-{}.json",
            std::process::id()
        ));
        three_way_artifact().save(&path).unwrap();
        let mut document: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        document["class_order"] = serde_json::json!(["breaking", "warning", "safe"]);
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        assert!(matches!(
            ThreeWayModelArtifact::load(&path),
            Err(ModelArtifactError::InvalidState { .. })
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_corrupt_or_incompatible_artifacts() {
        let path =
            std::env::temp_dir().join(format!("dcgaimodel-invalid-{}.json", std::process::id()));
        std::fs::write(&path, "not-json").unwrap();
        assert!(matches!(
            ModelArtifact::load(&path),
            Err(ModelArtifactError::Serialization { .. })
        ));
        std::fs::write(
            &path,
            r#"{"artifact_version":"unknown","model_version":"v1","feature_version":"dcg-features-v1","model_kind":"risk_score","threshold":0.5,"network":[],"scaler":{"means":[],"standard_deviations":[]},"training_metadata":{"optimizer":"sgd","epochs":1,"batch_size":1,"learning_rate":null,"seed":0,"training_samples":1,"validation_samples":0,"test_samples":0,"dataset_id":"x"}}"#,
        )
        .unwrap();
        assert!(matches!(
            ModelArtifact::load(&path),
            Err(ModelArtifactError::UnsupportedArtifactVersion { .. })
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_every_inference_critical_corruption_class() {
        let path =
            std::env::temp_dir().join(format!("dcgaimodel-corruption-{}.json", std::process::id()));
        let valid_path = std::env::temp_dir().join(format!(
            "dcgaimodel-corruption-source-{}.json",
            std::process::id()
        ));
        artifact().save(&valid_path).unwrap();
        let source = std::fs::read(&valid_path).unwrap();
        let valid: serde_json::Value = serde_json::from_slice(&source).unwrap();
        std::fs::remove_file(valid_path).unwrap();

        let mut cases = Vec::new();
        let mut wrong_feature_version = valid.clone();
        wrong_feature_version["feature_version"] =
            serde_json::Value::String("dcg-features-v2".to_owned());
        cases.push(wrong_feature_version);
        let mut empty_model_version = valid.clone();
        empty_model_version["model_version"] = serde_json::Value::String(String::new());
        cases.push(empty_model_version);
        let mut unknown_kind = valid.clone();
        unknown_kind["model_kind"] = serde_json::Value::String("unknown".to_owned());
        cases.push(unknown_kind);
        let mut unknown_activation = valid.clone();
        unknown_activation["network"][0]["activation"] =
            serde_json::Value::String("unknown".to_owned());
        cases.push(unknown_activation);
        let mut missing_layer = valid.clone();
        missing_layer["network"] = serde_json::json!([]);
        cases.push(missing_layer);
        let mut missing_weights = valid.clone();
        missing_weights["network"][0]["weights"] = serde_json::json!([]);
        cases.push(missing_weights);
        let mut wrong_bias_count = valid.clone();
        wrong_bias_count["network"][0]["biases"] = serde_json::json!([]);
        cases.push(wrong_bias_count);
        let mut invalid_threshold = valid.clone();
        invalid_threshold["threshold"] = serde_json::json!(1.1);
        cases.push(invalid_threshold);
        let mut invalid_scaler = valid.clone();
        invalid_scaler["scaler"]["standard_deviations"][0] = serde_json::json!(0.0);
        cases.push(invalid_scaler);
        let mut missing_metadata = valid.clone();
        missing_metadata
            .as_object_mut()
            .unwrap()
            .remove("training_metadata");
        cases.push(missing_metadata);

        for value in cases {
            std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
            assert!(ModelArtifact::load(&path).is_err());
        }
        // JSON cannot encode NaN/infinity. An out-of-range numeric token is
        // still rejected at the serialization boundary without a panic.
        std::fs::write(
            &path,
            String::from_utf8(source).unwrap().replacen(
                "\"threshold\": 0.5",
                "\"threshold\": 1e999",
                1,
            ),
        )
        .unwrap();
        assert!(matches!(
            ModelArtifact::load(&path),
            Err(ModelArtifactError::Serialization { .. })
        ));
        std::fs::remove_file(path).unwrap();
    }
}
