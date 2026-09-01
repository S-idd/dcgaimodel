//! Schema-pair feature extraction for the training-oriented V2 DCG contract.
//!
//! V2 records observable structural and policy context only. The oracle label
//! is deliberately not an input feature.

use super::{ContractFeatureExtractor, FeatureError};
use crate::linalg::Vector;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Stable feature schema for oracle-labelled schema-pair training data.
pub const DCG_FEATURE_V2_VERSION: &str = "dcg-features-v2";
/// Number of ordered V2 feature values.
pub const DCG_FEATURE_V2_COUNT: usize = 28;
/// Ordered names for the V2 raw structural and policy-context features.
pub const DCG_FEATURE_V2_NAMES: [&str; DCG_FEATURE_V2_COUNT] = [
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
    "policy_baseline",
    "policy_strict",
    "policy_relaxed",
    "policy_other",
];

/// Extracts V2 features from the real base/candidate schema pair and policy context.
#[derive(Debug, Clone, Copy, Default)]
pub struct SchemaChangeFeatureExtractor;

impl SchemaChangeFeatureExtractor {
    /// Creates a V2 schema-pair extractor.
    pub const fn new() -> Self {
        Self
    }

    /// Extracts a finite, label-free V2 feature vector.
    pub fn extract(
        &self,
        base: &str,
        candidate: &str,
        policy_pack: &str,
    ) -> Result<Vector, FeatureError> {
        let base =
            serde_json::from_str::<Value>(base).map_err(|error| FeatureError::SchemaJson {
                message: error.to_string(),
            })?;
        let candidate =
            serde_json::from_str::<Value>(candidate).map_err(|error| FeatureError::SchemaJson {
                message: error.to_string(),
            })?;
        let base_nodes = collect_properties(&base);
        let candidate_nodes = collect_properties(&candidate);
        let base_names = base_nodes.keys().cloned().collect::<BTreeSet<_>>();
        let candidate_names = candidate_nodes.keys().cloned().collect::<BTreeSet<_>>();
        let fields_added = candidate_names.difference(&base_names).count();
        let fields_removed = base_names.difference(&candidate_names).count();
        let shared = base_names
            .intersection(&candidate_names)
            .collect::<Vec<_>>();
        let fields_changed = shared
            .iter()
            .filter(|name| canonical(base_nodes[**name]) != canonical(candidate_nodes[**name]))
            .count();
        let mut type_changes = 0;
        let mut type_widening = 0;
        let mut type_narrowing = 0;
        let mut enum_values_added = 0;
        let mut enum_values_removed = 0;
        let mut enum_definitions_added = 0;
        let mut enum_definitions_removed = 0;
        let mut constraint_tightened = 0;
        let mut constraint_relaxed = 0;
        let mut additional_properties_tightened = 0;
        let mut additional_properties_relaxed = 0;
        for name in &shared {
            let before = base_nodes[*name];
            let after = candidate_nodes[*name];
            let before_types = type_set(before);
            let after_types = type_set(after);
            if before_types != after_types {
                type_changes += 1;
                if is_type_widening(&before_types, &after_types) {
                    type_widening += 1;
                }
                if is_type_widening(&after_types, &before_types) {
                    type_narrowing += 1;
                }
            }
            let before_enum = enum_values(before);
            let after_enum = enum_values(after);
            enum_values_added += after_enum.difference(&before_enum).count();
            enum_values_removed += before_enum.difference(&after_enum).count();
            if before_enum.is_empty() && !after_enum.is_empty() {
                enum_definitions_added += 1;
            }
            if !before_enum.is_empty() && after_enum.is_empty() {
                enum_definitions_removed += 1;
            }
            let (tight, relax) = constraint_delta(before, after);
            constraint_tightened += tight;
            constraint_relaxed += relax;
            match (
                before.get("additionalProperties"),
                after.get("additionalProperties"),
            ) {
                (Some(Value::Bool(false)), value) if value != Some(&Value::Bool(false)) => {
                    additional_properties_relaxed += 1
                }
                (value, Some(Value::Bool(false))) if value != Some(&Value::Bool(false)) => {
                    additional_properties_tightened += 1
                }
                _ => {}
            }
        }
        let base_required = required_paths(&base);
        let candidate_required = required_paths(&candidate);
        let policy = policy_features(policy_pack);
        let vector = Vector::new(vec![
            candidate_nodes.len() as f64,
            fields_added as f64,
            fields_removed as f64,
            fields_changed as f64,
            schema_depth(&candidate) as f64,
            candidate_required.len() as f64,
            candidate_required.difference(&base_required).count() as f64,
            base_required.difference(&candidate_required).count() as f64,
            type_changes as f64,
            type_widening as f64,
            type_narrowing as f64,
            enum_values_added as f64,
            enum_values_removed as f64,
            enum_definitions_added as f64,
            enum_definitions_removed as f64,
            constraint_tightened as f64,
            constraint_relaxed as f64,
            f64::from(changed(&base, &candidate, "if")),
            f64::from(changed(&base, &candidate, "oneOf")),
            f64::from(changed(&base, &candidate, "anyOf")),
            f64::from(changed(&base, &candidate, "allOf")),
            f64::from(changed(&base, &candidate, "not")),
            additional_properties_tightened as f64,
            additional_properties_relaxed as f64,
            policy.0,
            policy.1,
            policy.2,
            policy.3,
        ]);
        validate_feature_vector_v2(&vector)?;
        Ok(vector)
    }
}

/// Validates V2 values independently of the legacy V1 feature contract.
pub fn validate_feature_vector_v2(features: &Vector) -> Result<(), FeatureError> {
    if features.len() != DCG_FEATURE_V2_COUNT {
        return Err(FeatureError::FeatureCountMismatch {
            expected: DCG_FEATURE_V2_COUNT,
            actual: features.len(),
        });
    }
    for (index, value) in features.iter().enumerate() {
        if !value.is_finite() {
            return Err(FeatureError::NonFiniteFeature {
                field: DCG_FEATURE_V2_NAMES[index],
                value: *value,
            });
        }
        if *value < 0.0 {
            return Err(FeatureError::InvalidFeatureValue {
                field: DCG_FEATURE_V2_NAMES[index],
                value: *value,
            });
        }
    }
    Ok(())
}

/// Validates a vector against its explicitly persisted feature schema version.
pub fn validate_feature_vector_for_version(
    version: &str,
    features: &Vector,
) -> Result<(), FeatureError> {
    match version {
        super::DCG_FEATURE_VERSION => ContractFeatureExtractor::validate_feature_vector(features),
        DCG_FEATURE_V2_VERSION => validate_feature_vector_v2(features),
        super::DCG_FEATURE_V3_VERSION => super::validate_feature_vector_v3(features),
        super::DCG_FEATURE_V4_VERSION => super::validate_feature_vector_v4(features),
        super::DCG_FEATURE_V5_VERSION => super::validate_feature_vector_v5(features),
        super::DCG_FEATURE_V6_VERSION => super::validate_feature_vector_v6(features),
        _ => Err(FeatureError::SchemaJson {
            message: format!("unsupported feature version: {version}"),
        }),
    }
}

/// Returns the fixed vector dimension for a known feature schema version.
pub fn feature_count_for_version(version: &str) -> Option<usize> {
    match version {
        super::DCG_FEATURE_VERSION => Some(super::DCG_FEATURE_COUNT),
        DCG_FEATURE_V2_VERSION => Some(DCG_FEATURE_V2_COUNT),
        super::DCG_FEATURE_V3_VERSION => Some(super::DCG_FEATURE_V3_COUNT),
        super::DCG_FEATURE_V4_VERSION => Some(super::DCG_FEATURE_V4_COUNT),
        super::DCG_FEATURE_V5_VERSION => Some(super::DCG_FEATURE_V5_COUNT),
        super::DCG_FEATURE_V6_VERSION => Some(super::DCG_FEATURE_V6_COUNT),
        _ => None,
    }
}

fn collect_properties(value: &Value) -> BTreeMap<String, &Value> {
    let mut output = BTreeMap::new();
    collect_properties_at(value, "", &mut output);
    output
}
fn collect_properties_at<'a>(
    value: &'a Value,
    path: &str,
    output: &mut BTreeMap<String, &'a Value>,
) {
    let Some(object) = value.as_object() else {
        return;
    };
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        for (name, child) in properties {
            let child_path = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}.{name}")
            };
            output.insert(child_path.clone(), child);
            collect_properties_at(child, &child_path, output);
        }
    }
    if let Some(items) = object.get("items") {
        collect_properties_at(items, &format!("{path}[]"), output);
    }
}
fn required_paths(value: &Value) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    required_paths_at(value, "", &mut output);
    output
}
fn required_paths_at(value: &Value, path: &str, output: &mut BTreeSet<String>) {
    let Some(object) = value.as_object() else {
        return;
    };
    if let Some(required) = object.get("required").and_then(Value::as_array) {
        for name in required.iter().filter_map(Value::as_str) {
            output.insert(if path.is_empty() {
                name.to_owned()
            } else {
                format!("{path}.{name}")
            });
        }
    }
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        for (name, child) in properties {
            let child_path = if path.is_empty() {
                name.clone()
            } else {
                format!("{path}.{name}")
            };
            required_paths_at(child, &child_path, output);
        }
    }
}
fn type_set(value: &Value) -> BTreeSet<String> {
    match value.get("type") {
        Some(Value::String(value)) => [value.clone()].into_iter().collect(),
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => BTreeSet::new(),
    }
}
fn is_type_widening(before: &BTreeSet<String>, after: &BTreeSet<String>) -> bool {
    before.is_subset(after)
        || (before.len() == 1
            && after.len() == 1
            && before.contains("integer")
            && after.contains("number"))
}
fn enum_values(value: &Value) -> BTreeSet<String> {
    value
        .get("enum")
        .and_then(Value::as_array)
        .map(|values| values.iter().map(canonical).collect())
        .unwrap_or_default()
}
fn constraint_delta(before: &Value, after: &Value) -> (usize, usize) {
    const LOWER: [&str; 5] = [
        "minimum",
        "exclusiveMinimum",
        "minLength",
        "minItems",
        "minProperties",
    ];
    const UPPER: [&str; 5] = [
        "maximum",
        "exclusiveMaximum",
        "maxLength",
        "maxItems",
        "maxProperties",
    ];
    const OTHER: [&str; 5] = ["pattern", "format", "const", "uniqueItems", "multipleOf"];
    let mut tight = 0;
    let mut relax = 0;
    for key in LOWER {
        compare_constraint(
            before.get(key),
            after.get(key),
            true,
            &mut tight,
            &mut relax,
        );
    }
    for key in UPPER {
        compare_constraint(
            before.get(key),
            after.get(key),
            false,
            &mut tight,
            &mut relax,
        );
    }
    for key in OTHER {
        match (before.get(key), after.get(key)) {
            (None, Some(_)) => tight += 1,
            (Some(_), None) => relax += 1,
            (Some(left), Some(right)) if left != right => tight += 1,
            _ => {}
        }
    }
    (tight, relax)
}
fn compare_constraint(
    before: Option<&Value>,
    after: Option<&Value>,
    lower_is_tighter: bool,
    tight: &mut usize,
    relax: &mut usize,
) {
    match (
        before.and_then(Value::as_f64),
        after.and_then(Value::as_f64),
    ) {
        (None, Some(_)) => *tight += 1,
        (Some(_), None) => *relax += 1,
        (Some(left), Some(right)) if left != right => {
            if (right > left) == lower_is_tighter {
                *tight += 1
            } else {
                *relax += 1
            }
        }
        _ => {}
    }
}
fn changed(base: &Value, candidate: &Value, key: &str) -> bool {
    base.get(key).map(canonical) != candidate.get(key).map(canonical)
}
fn schema_depth(value: &Value) -> usize {
    match value {
        Value::Array(values) => 1 + values.iter().map(schema_depth).max().unwrap_or(0),
        Value::Object(values) => 1 + values.values().map(schema_depth).max().unwrap_or(0),
        _ => 0,
    }
}
fn policy_features(policy: &str) -> (f64, f64, f64, f64) {
    match policy {
        "baseline" => (1.0, 0.0, 0.0, 0.0),
        "strict" => (0.0, 1.0, 0.0, 0.0),
        "relaxed" => (0.0, 0.0, 1.0, 0.0),
        _ => (0.0, 0.0, 0.0, 1.0),
    }
}
fn canonical(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut values = map
                .iter()
                .map(|(k, v)| (k, canonical(v)))
                .collect::<Vec<_>>();
            values.sort_by(|a, b| a.0.cmp(b.0));
            format!(
                "{{{}}}",
                values
                    .into_iter()
                    .map(|(k, v)| format!("{k:?}:{v}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
        Value::Array(values) => format!(
            "[{}]",
            values.iter().map(canonical).collect::<Vec<_>>().join(",")
        ),
        _ => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extracts_policy_aware_structural_changes_without_a_label_feature() {
        let base = r#"{"type":"object","properties":{"kind":{"type":"string","enum":["a","b"]},"count":{"type":"integer"}},"required":["kind"]}"#;
        let candidate = r#"{"type":"object","properties":{"kind":{"type":"string","enum":["a","c"]},"count":{"type":"number","minimum":0},"added":{"type":"string"}},"required":["kind","count"],"if":{"properties":{"kind":{"const":"a"}}},"then":{"required":["added"]}}"#;
        let features = SchemaChangeFeatureExtractor::new()
            .extract(base, candidate, "strict")
            .unwrap();
        assert_eq!(features.len(), DCG_FEATURE_V2_COUNT);
        assert_eq!(features[1], 1.0);
        assert_eq!(features[6], 1.0);
        assert_eq!(features[9], 1.0);
        assert_eq!(features[11], 1.0);
        assert_eq!(features[12], 1.0);
        assert_eq!(features[17], 1.0);
        assert_eq!(features[25], 1.0);
    }
}
