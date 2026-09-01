use super::{ContractChange, ContractFeatures};
use crate::linalg::Vector;
use std::error::Error;
use std::fmt;

const MAX_EXACT_INTEGER: usize = 9_007_199_254_740_991;

/// Errors returned while validating or extracting DCG features.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureError {
    /// A JSON Schema source document could not be parsed for V2 extraction.
    SchemaJson { message: String },
    /// An approved policy-pack document could not be parsed or resolved.
    PolicyContext { message: String },
    /// A contract name was missing or whitespace-only.
    InvalidContract { reason: &'static str },
    /// A count cannot be represented exactly as an `f64` feature.
    UnsupportedMetadata { field: &'static str, value: usize },
    /// A produced feature was not finite.
    NonFiniteFeature { field: &'static str, value: f64 },
    /// A numeric feature vector did not contain the canonical number of values.
    FeatureCountMismatch { expected: usize, actual: usize },
    /// A numeric feature violates the canonical DCG feature schema.
    InvalidFeatureValue { field: &'static str, value: f64 },
}

impl fmt::Display for FeatureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaJson { message } => write!(f, "Invalid JSON Schema: {message}"),
            Self::PolicyContext { message } => write!(f, "Invalid policy context: {message}"),
            Self::InvalidContract { reason } => write!(f, "Invalid contract: {reason}"),
            Self::UnsupportedMetadata { field, value } => {
                write!(f, "Unsupported value for {field}: {value}")
            }
            Self::NonFiniteFeature { field, value } => {
                write!(f, "Non-finite feature {field}: {value}")
            }
            Self::FeatureCountMismatch { expected, actual } => {
                write!(
                    f,
                    "Feature count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidFeatureValue { field, value } => {
                write!(f, "Invalid value for feature {field}: {value}")
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

    /// Validates a canonical raw DCG feature vector before preprocessing or inference.
    ///
    /// Count-like values and semantic-version ordinals are non-negative. The
    /// compatibility score is exactly `0.0`, `0.5`, or `1.0`; all values must
    /// be finite. Missing values are unsupported and must be rejected rather
    /// than inferred or silently imputed.
    pub fn validate_feature_vector(features: &Vector) -> Result<(), FeatureError> {
        use crate::features::{DCG_FEATURE_COUNT, DCG_FEATURE_NAMES};

        if features.len() != DCG_FEATURE_COUNT {
            return Err(FeatureError::FeatureCountMismatch {
                expected: DCG_FEATURE_COUNT,
                actual: features.len(),
            });
        }
        for (index, value) in features.iter().enumerate() {
            let field = DCG_FEATURE_NAMES[index];
            if !value.is_finite() {
                return Err(FeatureError::NonFiniteFeature {
                    field,
                    value: *value,
                });
            }
            if index == 4 {
                if !(*value == 0.0 || *value == 0.5 || *value == 1.0) {
                    return Err(FeatureError::InvalidFeatureValue {
                        field,
                        value: *value,
                    });
                }
            } else if *value < 0.0 {
                return Err(FeatureError::InvalidFeatureValue {
                    field,
                    value: *value,
                });
            }
        }
        Ok(())
    }
}

impl FeatureExtractor<ContractChange> for ContractFeatureExtractor {
    fn extract(&self, input: &ContractChange) -> Result<Vector, FeatureError> {
        let features = self.extract_features(input)?.to_vector();
        Self::validate_feature_vector(&features)?;
        Ok(features)
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

    #[test]
    fn validates_the_canonical_feature_schema() {
        assert_eq!(
            ContractFeatureExtractor::validate_feature_vector(&Vector::new(vec![
                1.0, 0.0, 0.0, 0.0, 0.25, 1.0, 0.0, 0.0,
            ])),
            Err(FeatureError::InvalidFeatureValue {
                field: "compatibility_score",
                value: 0.25,
            })
        );
        assert!(matches!(
            ContractFeatureExtractor::validate_feature_vector(&Vector::new(vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])),
            Err(FeatureError::FeatureCountMismatch { .. })
        ));
        assert!(matches!(
            ContractFeatureExtractor::validate_feature_vector(&Vector::new(vec![
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                f64::NAN,
                0.0,
                0.0,
            ])),
            Err(FeatureError::NonFiniteFeature { .. })
        ));
    }
}
