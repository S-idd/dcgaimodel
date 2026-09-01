//! Error-directed V6 schema-pair features.
//!
//! V6 keeps V5's complete resolved policy context and adds only observable,
//! label-free facts that address the V5 challenge failures:
//!
//! - the actions of policy rules actually triggered by this transition;
//! - the structural subtype of a tightened constraint; and
//! - whether an optional root field is being added to an open or closed object.
//!
//! In particular, V6 never reads the oracle exit code, stdout, provenance
//! label, mutation name, or mutation variant.

use super::{
    FeatureError, POLICY_RULE_COUNT, PolicyFeatureContext, SchemaChangeFeatureV5Extractor,
};
use crate::linalg::Vector;
use serde_json::Value;
use std::collections::BTreeSet;

pub const DCG_FEATURE_V6_VERSION: &str = "dcg-features-v6";
pub const DCG_FEATURE_V6_COUNT: usize = 76;
pub const DCG_FEATURE_V6_NAMES: [&str; DCG_FEATURE_V6_COUNT] = [
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
    "active_rule_action_ignore",
    "active_rule_action_warning",
    "active_rule_action_breaking",
    "root_object_closed_before",
    "root_object_closed_after",
    "root_optional_fields_added",
    "root_optional_fields_added_to_closed_object",
    "constraint_numeric_tightened",
    "constraint_string_tightened",
    "constraint_array_tightened",
    "constraint_object_tightened",
    "constraint_lower_bound_tightened",
    "constraint_upper_bound_tightened",
    "constraint_pattern_tightened",
    "constraint_multiple_of_tightened",
    "constraint_unique_items_tightened",
    "constraint_compound_tightened",
];

/// Extracts V6 schema structure and resolved policy context.
#[derive(Debug, Clone, Copy, Default)]
pub struct SchemaChangeFeatureV6Extractor;

impl SchemaChangeFeatureV6Extractor {
    pub const fn new() -> Self {
        Self
    }

    /// Extracts label-free V6 features for a base/candidate schema pair.
    pub fn extract(
        &self,
        base: &str,
        candidate: &str,
        policy_context: PolicyFeatureContext,
    ) -> Result<Vector, FeatureError> {
        let v5 = SchemaChangeFeatureV5Extractor::new().extract(base, candidate, policy_context)?;
        let base = parse(base)?;
        let candidate = parse(candidate)?;
        let active_rules = active_rules(&v5);
        let active_actions = policy_context.active_rule_action_features(&active_rules);
        let optional = optional_field_context(&base, &candidate);
        let constraints = constraint_subtypes(&base, &candidate);

        let mut values = v5.iter().copied().collect::<Vec<_>>();
        values.extend(active_actions);
        values.extend(optional);
        values.extend(constraints);
        let vector = Vector::new(values);
        validate_feature_vector_v6(&vector)?;
        Ok(vector)
    }
}

/// Validates V6's fixed dimension and finite, non-negative representation.
pub fn validate_feature_vector_v6(features: &Vector) -> Result<(), FeatureError> {
    if features.len() != DCG_FEATURE_V6_COUNT {
        return Err(FeatureError::FeatureCountMismatch {
            expected: DCG_FEATURE_V6_COUNT,
            actual: features.len(),
        });
    }
    for (index, value) in features.iter().enumerate() {
        if !value.is_finite() {
            return Err(FeatureError::NonFiniteFeature {
                field: DCG_FEATURE_V6_NAMES[index],
                value: *value,
            });
        }
        if *value < 0.0 {
            return Err(FeatureError::InvalidFeatureValue {
                field: DCG_FEATURE_V6_NAMES[index],
                value: *value,
            });
        }
    }
    Ok(())
}

fn parse(schema: &str) -> Result<Value, FeatureError> {
    serde_json::from_str(schema).map_err(|error| FeatureError::SchemaJson {
        message: error.to_string(),
    })
}

fn active_rules(v5: &Vector) -> [bool; POLICY_RULE_COUNT] {
    // The first 24 V5 values are preserved V2 structural facts. The final
    // five values identify root schema restrictions added by the generator.
    [
        v5[2] > 0.0,
        v5[8] > 0.0,
        v5[6] > 0.0,
        v5[12] > 0.0 || v5[14] > 0.0,
        v5[11] > 0.0 || v5[13] > 0.0,
        v5[15] > 0.0,
        v5[17] > 0.0,
        v5[22] > 0.0
            || v5[54] > 0.0
            || v5[55] > 0.0
            || v5[56] > 0.0
            || v5[57] > 0.0
            || v5[58] > 0.0,
    ]
}

fn optional_field_context(base: &Value, candidate: &Value) -> [f64; 4] {
    let base_closed = root_object_is_closed(base);
    let candidate_closed = root_object_is_closed(candidate);
    let base_properties = root_property_names(base);
    let candidate_properties = candidate.get("properties").and_then(Value::as_object);
    let required = root_required_names(candidate);
    let optional_added = candidate_properties
        .map(|properties| {
            properties
                .keys()
                .filter(|name| !base_properties.contains(*name) && !required.contains(*name))
                .count()
        })
        .unwrap_or_default();
    [
        f64::from(base_closed),
        f64::from(candidate_closed),
        optional_added as f64,
        if candidate_closed {
            optional_added as f64
        } else {
            0.0
        },
    ]
}

fn root_object_is_closed(value: &Value) -> bool {
    value.get("additionalProperties") == Some(&Value::Bool(false))
}

fn root_property_names(value: &Value) -> BTreeSet<String> {
    value
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| properties.keys().cloned().collect())
        .unwrap_or_default()
}

fn root_required_names(value: &Value) -> BTreeSet<String> {
    value
        .get("required")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn constraint_subtypes(base: &Value, candidate: &Value) -> [f64; 10] {
    let Some(base_properties) = base.get("properties").and_then(Value::as_object) else {
        return [0.0; 10];
    };
    let Some(candidate_properties) = candidate.get("properties").and_then(Value::as_object) else {
        return [0.0; 10];
    };
    let mut values = [0.0; 10];
    for (name, before) in base_properties {
        let Some(after) = candidate_properties.get(name) else {
            continue;
        };
        let changes = tightened_constraint_keywords(before, after);
        if changes.is_empty() {
            continue;
        }
        match after.get("type").and_then(Value::as_str) {
            Some("integer") | Some("number") => values[0] += 1.0,
            Some("string") => values[1] += 1.0,
            Some("array") => values[2] += 1.0,
            Some("object") => values[3] += 1.0,
            _ => {}
        }
        for change in &changes {
            match change {
                ConstraintKind::Lower => values[4] += 1.0,
                ConstraintKind::Upper => values[5] += 1.0,
                ConstraintKind::Pattern => values[6] += 1.0,
                ConstraintKind::MultipleOf => values[7] += 1.0,
                ConstraintKind::UniqueItems => values[8] += 1.0,
                ConstraintKind::Other => {}
            }
        }
        if changes.len() > 1 {
            values[9] += 1.0;
        }
    }
    values
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstraintKind {
    Lower,
    Upper,
    Pattern,
    MultipleOf,
    UniqueItems,
    Other,
}

fn tightened_constraint_keywords(before: &Value, after: &Value) -> Vec<ConstraintKind> {
    let mut changes = Vec::new();
    for key in [
        "minimum",
        "exclusiveMinimum",
        "minLength",
        "minItems",
        "minProperties",
    ] {
        if numeric_constraint_tightened(before.get(key), after.get(key), true) {
            changes.push(ConstraintKind::Lower);
        }
    }
    for key in [
        "maximum",
        "exclusiveMaximum",
        "maxLength",
        "maxItems",
        "maxProperties",
    ] {
        if numeric_constraint_tightened(before.get(key), after.get(key), false) {
            changes.push(ConstraintKind::Upper);
        }
    }
    if non_numeric_constraint_tightened(before.get("pattern"), after.get("pattern")) {
        changes.push(ConstraintKind::Pattern);
    }
    if non_numeric_constraint_tightened(before.get("multipleOf"), after.get("multipleOf")) {
        changes.push(ConstraintKind::MultipleOf);
    }
    if non_numeric_constraint_tightened(before.get("uniqueItems"), after.get("uniqueItems")) {
        changes.push(ConstraintKind::UniqueItems);
    }
    for key in ["format", "const"] {
        if non_numeric_constraint_tightened(before.get(key), after.get(key)) {
            changes.push(ConstraintKind::Other);
        }
    }
    changes
}

fn numeric_constraint_tightened(
    before: Option<&Value>,
    after: Option<&Value>,
    lower: bool,
) -> bool {
    match (
        before.and_then(Value::as_f64),
        after.and_then(Value::as_f64),
    ) {
        (None, Some(_)) => true,
        (Some(left), Some(right)) if left != right => (right > left) == lower,
        _ => false,
    }
}

fn non_numeric_constraint_tightened(before: Option<&Value>, after: Option<&Value>) -> bool {
    match (before, after) {
        (None, Some(_)) => true,
        (Some(left), Some(right)) => left != right,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{ApprovedPolicyContexts, DCG_FEATURE_V5_COUNT};

    #[test]
    fn exposes_active_constraint_action_and_subtype_without_a_label() {
        let contexts = ApprovedPolicyContexts::from_json(
            r#"{"packs":{"ignore-constraint":{"rules":{"CONSTRAINT_TIGHTENED":"IGNORE"}}}}"#,
            &["ignore-constraint".to_owned()],
        )
        .unwrap();
        let base = r#"{"type":"object","properties":{"amount":{"type":"number"}}}"#;
        let candidate = r#"{"type":"object","properties":{"amount":{"type":"number","minimum":0,"multipleOf":2}}}"#;
        let features = SchemaChangeFeatureV6Extractor::new()
            .extract(base, candidate, contexts.get("ignore-constraint").unwrap())
            .unwrap();

        assert_eq!(features.len(), DCG_FEATURE_V6_COUNT);
        assert_eq!(features[DCG_FEATURE_V5_COUNT], 1.0);
        assert_eq!(features[DCG_FEATURE_V5_COUNT + 1], 0.0);
        assert_eq!(features[DCG_FEATURE_V5_COUNT + 2], 0.0);
        assert_eq!(features[DCG_FEATURE_V5_COUNT + 7], 1.0);
        assert_eq!(features[DCG_FEATURE_V5_COUNT + 11], 1.0);
        assert_eq!(features[DCG_FEATURE_V5_COUNT + 14], 1.0);
        assert_eq!(features[DCG_FEATURE_V5_COUNT + 16], 1.0);
    }

    #[test]
    fn distinguishes_optional_additions_to_open_and_closed_objects() {
        let contexts = ApprovedPolicyContexts::from_json(
            r#"{"packs":{"baseline":{}}}"#,
            &["baseline".to_owned()],
        )
        .unwrap();
        let open = r#"{"type":"object","properties":{"id":{"type":"string"}}}"#;
        let closed = r#"{"type":"object","additionalProperties":false,"properties":{"id":{"type":"string"}}}"#;
        let added = r#"{"type":"object","additionalProperties":false,"properties":{"id":{"type":"string"},"note":{"type":"string"}}}"#;
        let context = contexts.get("baseline").unwrap();
        let open_features = SchemaChangeFeatureV6Extractor::new()
            .extract(open, added, context)
            .unwrap();
        let closed_features = SchemaChangeFeatureV6Extractor::new()
            .extract(closed, added, context)
            .unwrap();

        assert_eq!(open_features[DCG_FEATURE_V5_COUNT + 3], 0.0);
        assert_eq!(open_features[DCG_FEATURE_V5_COUNT + 5], 1.0);
        assert_eq!(open_features[DCG_FEATURE_V5_COUNT + 6], 1.0);
        assert_eq!(closed_features[DCG_FEATURE_V5_COUNT + 3], 1.0);
        assert_eq!(closed_features[DCG_FEATURE_V5_COUNT + 5], 1.0);
        assert_eq!(closed_features[DCG_FEATURE_V5_COUNT + 6], 1.0);
    }
}
