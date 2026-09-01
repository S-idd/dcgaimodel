//! V4 features replace policy-name identity with resolved policy semantics.
//!
//! The policy action is parsed directly from the approved policy-pack JSON,
//! never copied from an oracle exit code or compatibility label. This lets a
//! held-out policy be represented by what it declares rather than an unseen
//! one-hot policy name.

use super::{FeatureError, PolicyFeatureContext, SchemaChangeFeatureV3Extractor};
use crate::linalg::Vector;

pub const DCG_FEATURE_V4_VERSION: &str = "dcg-features-v4";
pub const DCG_FEATURE_V4_COUNT: usize = 38;
pub const DCG_FEATURE_V4_NAMES: [&str; DCG_FEATURE_V4_COUNT] = [
    "field_count",
    "fields_added",
    "fields_removed",
    "fields_changed",
    "schema_depth",
    "required_field_count",
    "required_fields_added",
    "required_fields_removed",
    "type_changes",
    "type_widening",
    "type_narrowing",
    "enum_values_added",
    "enum_values_removed",
    "enum_definitions_added",
    "enum_definitions_removed",
    "constraint_tightened",
    "constraint_relaxed",
    "conditional_changed",
    "one_of_changed",
    "any_of_changed",
    "all_of_changed",
    "not_changed",
    "additional_properties_tightened",
    "additional_properties_relaxed",
    "policy_enum_value_added_ignore",
    "policy_enum_value_added_warning",
    "policy_enum_value_added_breaking",
    "root_fields_added_string",
    "root_fields_added_numeric",
    "root_fields_added_boolean",
    "root_fields_added_array",
    "root_fields_added_object",
    "root_fields_added_other",
    "root_additional_properties_tightened",
    "root_property_names_changed",
    "root_dependent_required_changed",
    "root_dependent_schemas_changed",
    "root_schema_applicators_changed",
];

/// Extracts V4 schema structure plus resolved policy configuration context.
#[derive(Debug, Clone, Copy, Default)]
pub struct SchemaChangeFeatureV4Extractor;

impl SchemaChangeFeatureV4Extractor {
    pub const fn new() -> Self {
        Self
    }

    /// Extracts label-free V4 features for one schema pair and declared policy.
    pub fn extract(
        &self,
        base: &str,
        candidate: &str,
        policy_context: PolicyFeatureContext,
    ) -> Result<Vector, FeatureError> {
        // V3's name-dependent policy values are removed. The arbitrary pack
        // name is therefore deliberately not represented in V4.
        let mut values = SchemaChangeFeatureV3Extractor::new()
            .extract(base, candidate, "policy-name-not-a-feature")?
            .iter()
            .copied()
            .collect::<Vec<_>>();
        values.drain(24..28);
        values.splice(24..24, policy_context.enum_value_added_features());
        let vector = Vector::new(values);
        validate_feature_vector_v4(&vector)?;
        Ok(vector)
    }
}

/// Validates V4's fixed dimension and finite, non-negative representation.
pub fn validate_feature_vector_v4(features: &Vector) -> Result<(), FeatureError> {
    if features.len() != DCG_FEATURE_V4_COUNT {
        return Err(FeatureError::FeatureCountMismatch {
            expected: DCG_FEATURE_V4_COUNT,
            actual: features.len(),
        });
    }
    for (index, value) in features.iter().enumerate() {
        if !value.is_finite() {
            return Err(FeatureError::NonFiniteFeature {
                field: DCG_FEATURE_V4_NAMES[index],
                value: *value,
            });
        }
        if *value < 0.0 {
            return Err(FeatureError::InvalidFeatureValue {
                field: DCG_FEATURE_V4_NAMES[index],
                value: *value,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::ApprovedPolicyContexts;

    #[test]
    fn replaces_policy_name_with_declared_enum_action() {
        let contexts = ApprovedPolicyContexts::from_json(
            r#"{"defaultPack":"baseline","packs":{"baseline":{"rules":{"ENUM_VALUE_ADDED":"WARNING"}},"strict":{"rules":{"ENUM_VALUE_ADDED":"BREAKING"}}}}"#,
            &["baseline".to_owned(), "strict".to_owned()],
        )
        .unwrap();
        let base = r#"{"type":"object","properties":{"status":{"type":"string","enum":["old"]}}}"#;
        let candidate =
            r#"{"type":"object","properties":{"status":{"type":"string","enum":["old","new"]}}}"#;
        let baseline = SchemaChangeFeatureV4Extractor::new()
            .extract(base, candidate, contexts.get("baseline").unwrap())
            .unwrap();
        let strict = SchemaChangeFeatureV4Extractor::new()
            .extract(base, candidate, contexts.get("strict").unwrap())
            .unwrap();
        assert_eq!(baseline.len(), DCG_FEATURE_V4_COUNT);
        assert_eq!(
            &baseline.iter().copied().collect::<Vec<_>>()[24..27],
            &[0.0, 1.0, 0.0]
        );
        assert_eq!(
            &strict.iter().copied().collect::<Vec<_>>()[24..27],
            &[0.0, 0.0, 1.0]
        );
    }
}
