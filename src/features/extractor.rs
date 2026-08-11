use super::{ContractChange, ContractFeatures};
use crate::linalg::Vector;
use std::error::Error;
use std::fmt;

const MAX_EXACT_INTEGER: usize = 9_007_199_254_740_991;

/// Errors returned while validating or extracting DCG features.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureError {
    /// A contract name was missing or whitespace-only.
    InvalidContract { reason: &'static str },
    /// A count cannot be represented exactly as an `f64` feature.
    UnsupportedMetadata { field: &'static str, value: usize },
    /// A produced feature was not finite.
    NonFiniteFeature { field: &'static str, value: f64 },
}

impl fmt::Display for FeatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidContract { reason } => write!(f, "Invalid contract: {reason}"),
            Self::UnsupportedMetadata { field, value } => {
                write!(f, "Unsupported value for {field}: {value}")
            }
            Self::NonFiniteFeature { field, value } => {
                write!(f, "Non-finite feature {field}: {value}")
            }
        }
    }
}

impl Error for FeatureError {}

/// Converts a domain input into a validated numerical feature vector.
pub trait FeatureExtractor<T> {
    /// Extracts one deterministic, finite feature vector from `input`.
    fn extract(&self, input: &T) -> Result<Vector, FeatureError>;
}

/// Extracts the stable DCG feature representation from [`ContractChange`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ContractFeatureExtractor;

impl ContractFeatureExtractor {
    /// Creates a contract feature extractor.
    pub const fn new() -> Self {
        Self
    }

    /// Extracts the named features before converting them to a vector.
    pub fn extract_features(
        &self,
        contract: &ContractChange,
    ) -> Result<ContractFeatures, FeatureError> {
        if contract.contract_name.trim().is_empty() {
            return Err(FeatureError::InvalidContract {
                reason: "contract name must not be empty",
            });
        }

        for (field, value) in [
            ("number_of_fields", contract.number_of_fields),
            ("fields_added", contract.fields_added),
            ("fields_removed", contract.fields_removed),
            ("data_type_changes", contract.data_type_changes),
            ("breaking_changes", contract.breaking_changes),
            ("dependent_consumers", contract.metadata.dependent_consumers),
        ] {
            if value > MAX_EXACT_INTEGER {
                return Err(FeatureError::UnsupportedMetadata { field, value });
            }
        }

        let features = ContractFeatures {
            number_of_fields: contract.number_of_fields as f64,
            fields_added: contract.fields_added as f64,
            fields_removed: contract.fields_removed as f64,
            data_type_changes: contract.data_type_changes as f64,
            compatibility_status: contract.compatibility_status.as_feature_value(),
            schema_version: contract.schema_version.as_feature_value(),
            breaking_changes: contract.breaking_changes as f64,
            dependent_consumers: contract.metadata.dependent_consumers as f64,
        };

        for (field, value) in [
            ("number_of_fields", features.number_of_fields),
            ("fields_added", features.fields_added),
            ("fields_removed", features.fields_removed),
            ("data_type_changes", features.data_type_changes),
            ("compatibility_status", features.compatibility_status),
            ("schema_version", features.schema_version),
            ("breaking_changes", features.breaking_changes),
            ("dependent_consumers", features.dependent_consumers),
        ] {
            if !value.is_finite() {
                return Err(FeatureError::NonFiniteFeature { field, value });
            }
        }

        Ok(features)
    }
}

impl FeatureExtractor<ContractChange> for ContractFeatureExtractor {
    fn extract(&self, input: &ContractChange) -> Result<Vector, FeatureError> {
        Ok(self.extract_features(input)?.to_vector())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{CompatibilityStatus, ContractMetadata, SemanticVersion};

    fn contract() -> ContractChange {
        ContractChange {
            contract_name: "orders".to_owned(),
            number_of_fields: 8,
            fields_added: 2,
            fields_removed: 1,
            data_type_changes: 1,
            compatibility_status: CompatibilityStatus::Incompatible,
            schema_version: SemanticVersion::new(2, 4, 3),
            breaking_changes: 2,
            metadata: ContractMetadata {
                dependent_consumers: 5,
            },
        }
    }

    #[test]
    fn extracts_documented_feature_order() {
        let vector = ContractFeatureExtractor::new()
            .extract(&contract())
            .unwrap();

        assert_eq!(
            vector,
            Vector::new(vec![8.0, 2.0, 1.0, 1.0, 1.0, 2_004_003.0, 2.0, 5.0])
        );
    }

    #[test]
    fn feature_extraction_is_deterministic() {
        let extractor = ContractFeatureExtractor::new();
        assert_eq!(
            extractor.extract(&contract()),
            extractor.extract(&contract())
        );
    }

    #[test]
    fn rejects_empty_contract_name() {
        let mut input = contract();
        input.contract_name = "  ".to_owned();
        assert!(matches!(
            ContractFeatureExtractor::new().extract(&input),
            Err(FeatureError::InvalidContract { .. })
        ));
    }

    #[test]
    fn compatibility_and_version_encodings_are_documented() {
        assert_eq!(CompatibilityStatus::Compatible.as_feature_value(), 0.0);
        assert_eq!(CompatibilityStatus::Unknown.as_feature_value(), 0.5);
        assert_eq!(CompatibilityStatus::Incompatible.as_feature_value(), 1.0);
        assert_eq!(
            SemanticVersion::new(1, 2, 3).as_feature_value(),
            1_002_003.0
        );
    }
}
