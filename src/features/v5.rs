//! V5 schema-pair features with complete resolved policy-rule semantics.
//!
//! V5 removes all policy-name identity and encodes the pinned JAR's effective
//! IGNORE/WARNING/BREAKING action for each supported rule. The policy JSON is
//! the sole source for these features; no oracle result or label is read.

use super::{
    FeatureError, POLICY_RULE_ACTION_FEATURE_COUNT, PolicyFeatureContext,
    SchemaChangeFeatureV3Extractor,
};
use crate::linalg::Vector;

pub const DCG_FEATURE_V5_VERSION: &str = "dcg-features-v5";
pub const DCG_FEATURE_V5_COUNT: usize = 59;
pub const DCG_FEATURE_V5_NAMES: [&str; DCG_FEATURE_V5_COUNT] = [
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
    "policy_field_removed_ignore",
    "policy_field_removed_warning",
    "policy_field_removed_breaking",
    "policy_field_type_changed_ignore",
    "policy_field_type_changed_warning",
    "policy_field_type_changed_breaking",
    "policy_required_field_added_ignore",
    "policy_required_field_added_warning",
    "policy_required_field_added_breaking",
    "policy_enum_value_removed_ignore",
    "policy_enum_value_removed_warning",
    "policy_enum_value_removed_breaking",
    "policy_enum_value_added_ignore",
    "policy_enum_value_added_warning",
    "policy_enum_value_added_breaking",
    "policy_constraint_tightened_ignore",
    "policy_constraint_tightened_warning",
    "policy_constraint_tightened_breaking",
    "policy_conditional_restriction_added_ignore",
    "policy_conditional_restriction_added_warning",
    "policy_conditional_restriction_added_breaking",
    "policy_schema_restriction_added_ignore",
    "policy_schema_restriction_added_warning",
    "policy_schema_restriction_added_breaking",
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

/// Extracts V5 schema structure plus all resolved policy rule actions.
#[derive(Debug, Clone, Copy, Default)]
pub struct SchemaChangeFeatureV5Extractor;

impl SchemaChangeFeatureV5Extractor {
    pub const fn new() -> Self {
        Self
    }

    /// Extracts label-free V5 features for one schema pair and declared policy.
    pub fn extract(
        &self,
        base: &str,
        candidate: &str,
        policy_context: PolicyFeatureContext,
    ) -> Result<Vector, FeatureError> {
        // Remove V3's arbitrary policy-name one-hot fields, preserving only
        // structural schema signals around the full policy-action vector.
        let mut values = SchemaChangeFeatureV3Extractor::new()
            .extract(base, candidate, "policy-name-not-a-feature")?
            .iter()
            .copied()
            .collect::<Vec<_>>();
        // Explicitly drop the splice iterator now: `Vec::splice` performs
        // insertion on iterator drop, and the final vector must contain the
        // supplied context before it is validated or persisted.
        let policy_actions = policy_context.rule_action_features();
        drop(values.splice(24..28, policy_actions));
        let vector = Vector::new(values);
        validate_feature_vector_v5(&vector)?;
        Ok(vector)
    }
}

/// Validates V5's fixed dimension and finite, non-negative representation.
pub fn validate_feature_vector_v5(features: &Vector) -> Result<(), FeatureError> {
    if features.len() != DCG_FEATURE_V5_COUNT {
        return Err(FeatureError::FeatureCountMismatch {
            expected: DCG_FEATURE_V5_COUNT,
            actual: features.len(),
        });
    }
    for (index, value) in features.iter().enumerate() {
        if !value.is_finite() {
            return Err(FeatureError::NonFiniteFeature {
                field: DCG_FEATURE_V5_NAMES[index],
                value: *value,
            });
        }
        if *value < 0.0 {
            return Err(FeatureError::InvalidFeatureValue {
                field: DCG_FEATURE_V5_NAMES[index],
                value: *value,
            });
        }
    }
    Ok(())
}

const _: [(); POLICY_RULE_ACTION_FEATURE_COUNT] = [(); 24];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::ApprovedPolicyContexts;

    #[test]
    fn replaces_policy_name_with_all_declared_rule_actions() {
        let contexts = ApprovedPolicyContexts::from_json(
            r#"{"packs":{"baseline":{"rules":{"ENUM_VALUE_ADDED":"WARNING"}},"custom":{"rules":{"FIELD_REMOVED":"IGNORE","CONSTRAINT_TIGHTENED":"WARNING"}}}}"#,
            &["baseline".to_owned(), "custom".to_owned()],
        )
        .unwrap();
        let base = r#"{"type":"object","properties":{"status":{"type":"string","enum":["old"]}}}"#;
        let candidate =
            r#"{"type":"object","properties":{"status":{"type":"string","enum":["old","new"]}}}"#;
        let baseline = SchemaChangeFeatureV5Extractor::new()
            .extract(base, candidate, contexts.get("baseline").unwrap())
            .unwrap();
        let custom = SchemaChangeFeatureV5Extractor::new()
            .extract(base, candidate, contexts.get("custom").unwrap())
            .unwrap();
        assert_eq!(baseline.len(), DCG_FEATURE_V5_COUNT);
        assert_eq!(
            &contexts.get("custom").unwrap().rule_action_features()[15..18],
            &[0.0, 1.0, 0.0]
        );
        assert_eq!(
            &custom.iter().copied().collect::<Vec<_>>()[24..48],
            &contexts.get("custom").unwrap().rule_action_features()
        );
        assert_eq!(
            &baseline.iter().copied().collect::<Vec<_>>()[24..48],
            &[
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            ]
        );
    }
}
