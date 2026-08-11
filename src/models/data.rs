use crate::dataset::{Dataset, DatasetError};
use crate::features::{ContractChange, ContractFeatureExtractor, FeatureError, FeatureExtractor};
use crate::linalg::Vector;
use crate::prediction::PredictionKind;
use std::error::Error;
use std::fmt;

/// Supervised labels for a changed contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContractLabels {
    /// `0.0` for non-breaking and `1.0` for breaking.
    pub breaking_change: f64,
    /// `0.0` for compatible and `1.0` for incompatible.
    pub incompatible: f64,
    /// Normalized risk from `0.0` (lowest) to `1.0` (highest).
    pub risk_score: f64,
}

impl ContractLabels {
    /// Validates the binary labels and normalized risk score.
    pub fn validate(&self) -> Result<(), DcgDatasetError> {
        for (name, value) in [
            ("breaking_change", self.breaking_change),
            ("incompatible", self.incompatible),
        ] {
            if !value.is_finite() || !(value == 0.0 || value == 1.0) {
                return Err(DcgDatasetError::InvalidLabel { name, value });
            }
        }
        if !self.risk_score.is_finite() || !(0.0..=1.0).contains(&self.risk_score) {
            return Err(DcgDatasetError::InvalidLabel {
                name: "risk_score",
                value: self.risk_score,
            });
        }
        Ok(())
    }

    fn target_for(&self, kind: PredictionKind) -> Vector {
        let value = match kind {
            PredictionKind::BreakingChange => self.breaking_change,
            PredictionKind::Compatibility => self.incompatible,
            PredictionKind::RiskScore => self.risk_score,
        };
        Vector::new(vec![value])
    }
}

/// One DCG training or evaluation example.
#[derive(Debug, Clone, PartialEq)]
pub struct ContractExample {
    /// Domain input used for deterministic feature extraction.
    pub contract: ContractChange,
    /// Ground-truth labels for the three supported tasks.
    pub labels: ContractLabels,
}

/// Errors returned while converting DCG examples into the generic dataset.
#[derive(Debug, Clone, PartialEq)]
pub enum DcgDatasetError {
    /// No examples were supplied.
    EmptyExamples,
    /// Contract feature extraction failed.
    Feature(FeatureError),
    /// A supplied model target was invalid.
    InvalidLabel { name: &'static str, value: f64 },
    /// The reused generic dataset rejected generated data.
    Dataset(DatasetError),
}

impl fmt::Display for DcgDatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyExamples => write!(f, "At least one contract example is required."),
            Self::Feature(error) => write!(f, "Feature extraction error: {error}"),
            Self::InvalidLabel { name, value } => write!(f, "Invalid {name} label: {value}"),
            Self::Dataset(error) => write!(f, "Dataset error: {error}"),
        }
    }
}

impl Error for DcgDatasetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Feature(error) => Some(error),
            Self::Dataset(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FeatureError> for DcgDatasetError {
    fn from(error: FeatureError) -> Self {
        Self::Feature(error)
    }
}
impl From<DatasetError> for DcgDatasetError {
    fn from(error: DatasetError) -> Self {
        Self::Dataset(error)
    }
}

/// Extracts DCG features and creates the project's existing [`Dataset`].
pub fn build_dataset(
    examples: &[ContractExample],
    kind: PredictionKind,
) -> Result<Dataset, DcgDatasetError> {
    if examples.is_empty() {
        return Err(DcgDatasetError::EmptyExamples);
    }
    let extractor = ContractFeatureExtractor::new();
    let mut features = Vec::with_capacity(examples.len());
    let mut targets = Vec::with_capacity(examples.len());
    for example in examples {
        example.labels.validate()?;
        features.push(extractor.extract(&example.contract)?);
        targets.push(example.labels.target_for(kind));
    }
    Ok(Dataset::new(features, targets)?)
}

/// Returns a deterministic, synthetic DCG dataset for development and tests.
///
/// Labels are rule-based examples, not evidence of real-world production
/// behaviour. Trained model outputs are always produced by the neural network.
pub fn synthetic_examples() -> Vec<ContractExample> {
    use crate::features::{CompatibilityStatus, ContractMetadata, SemanticVersion};
    let example = |name: &str,
                   fields: usize,
                   added: usize,
                   removed: usize,
                   type_changes: usize,
                   status: CompatibilityStatus,
                   breaking: usize,
                   consumers: usize,
                   labels: ContractLabels| ContractExample {
        contract: ContractChange {
            contract_name: name.to_owned(),
            number_of_fields: fields,
            fields_added: added,
            fields_removed: removed,
            data_type_changes: type_changes,
            compatibility_status: status,
            schema_version: SemanticVersion::new(1, 0, 0),
            breaking_changes: breaking,
            metadata: ContractMetadata {
                dependent_consumers: consumers,
            },
        },
        labels,
    };
    vec![
        example(
            "orders-no-change",
            8,
            0,
            0,
            0,
            CompatibilityStatus::Compatible,
            0,
            2,
            ContractLabels {
                breaking_change: 0.0,
                incompatible: 0.0,
                risk_score: 0.0,
            },
        ),
        example(
            "orders-safe-addition",
            9,
            1,
            0,
            0,
            CompatibilityStatus::Compatible,
            0,
            3,
            ContractLabels {
                breaking_change: 0.0,
                incompatible: 0.0,
                risk_score: 0.15,
            },
        ),
        example(
            "orders-removed-field",
            7,
            0,
            1,
            0,
            CompatibilityStatus::Incompatible,
            1,
            4,
            ContractLabels {
                breaking_change: 1.0,
                incompatible: 1.0,
                risk_score: 0.65,
            },
        ),
        example(
            "orders-type-change",
            8,
            0,
            0,
            1,
            CompatibilityStatus::Incompatible,
            1,
            5,
            ContractLabels {
                breaking_change: 1.0,
                incompatible: 1.0,
                risk_score: 0.7,
            },
        ),
        example(
            "orders-multiple-breaking",
            6,
            0,
            2,
            2,
            CompatibilityStatus::Incompatible,
            3,
            8,
            ContractLabels {
                breaking_change: 1.0,
                incompatible: 1.0,
                risk_score: 1.0,
            },
        ),
        example(
            "orders-unknown-impact",
            8,
            0,
            0,
            0,
            CompatibilityStatus::Unknown,
            0,
            1,
            ContractLabels {
                breaking_change: 0.0,
                incompatible: 0.0,
                risk_score: 0.25,
            },
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn examples_flow_from_contract_to_generic_dataset() {
        let dataset = build_dataset(&synthetic_examples(), PredictionKind::BreakingChange).unwrap();
        assert_eq!(dataset.len(), 6);
        assert_eq!(dataset.feature_size(), 8);
        assert_eq!(dataset.target_size(), 1);
        assert_eq!(dataset.targets()[2], Vector::new(vec![1.0]));
    }

    #[test]
    fn validates_labels_and_empty_example_sets() {
        assert_eq!(
            build_dataset(&[], PredictionKind::RiskScore),
            Err(DcgDatasetError::EmptyExamples)
        );
        let mut examples = synthetic_examples();
        examples[0].labels.risk_score = f64::INFINITY;
        assert_eq!(
            build_dataset(&examples, PredictionKind::RiskScore),
            Err(DcgDatasetError::InvalidLabel {
                name: "risk_score",
                value: f64::INFINITY
            })
        );
    }
}
