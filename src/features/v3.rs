//! V3 schema-pair features for counterfactual mutation generalization.
//!
//! V3 preserves the V2 structural/policy features and adds observable signals
//! for newly added field types and root-level schema restrictions. These are
//! required to distinguish the richer hard-family candidates produced by the
//! generator; no oracle outcome is included as a feature.

use super::{FeatureError, SchemaChangeFeatureExtractor};
use crate::linalg::Vector;
use serde_json::Value;
use std::collections::BTreeSet;

pub const DCG_FEATURE_V3_VERSION: &str = "dcg-features-v3";
pub const DCG_FEATURE_V3_COUNT: usize = 39;
pub const DCG_FEATURE_V3_NAMES: [&str; DCG_FEATURE_V3_COUNT] = [
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

/// Extracts V3 features from one schema transition and its policy context.
#[derive(Debug, Clone, Copy, Default)]
pub struct SchemaChangeFeatureV3Extractor;

impl SchemaChangeFeatureV3Extractor {
    pub const fn new() -> Self {
        Self
    }

    pub fn extract(
        &self,
        base: &str,
        candidate: &str,
        policy_pack: &str,
    ) -> Result<Vector, FeatureError> {
        let mut values = SchemaChangeFeatureExtractor::new()
            .extract(base, candidate, policy_pack)?
            .iter()
            .copied()
            .collect::<Vec<_>>();
        let base = parse(base)?;
        let candidate = parse(candidate)?;
        let base_properties = root_property_names(&base);
        let candidate_properties = candidate.get("properties").and_then(Value::as_object);
        let mut added_types = [0usize; 6];
        if let Some(properties) = candidate_properties {
            for (name, schema) in properties {
                if !base_properties.contains(name) {
                    added_types[type_bucket(schema)] += 1;
                }
            }
        }
        values.extend(added_types.map(|value| value as f64));
        values.extend([
            f64::from(root_additional_properties_tightened(&base, &candidate)),
            f64::from(root_keyword_changed(&base, &candidate, "propertyNames")),
            f64::from(root_keyword_changed(&base, &candidate, "dependentRequired")),
            f64::from(root_keyword_changed(&base, &candidate, "dependentSchemas")),
            f64::from(
                root_keyword_changed(&base, &candidate, "patternProperties")
                    || root_keyword_changed(&base, &candidate, "not")
                    || root_keyword_changed(&base, &candidate, "unevaluatedProperties"),
            ),
        ]);
        let vector = Vector::new(values);
        validate_feature_vector_v3(&vector)?;
        Ok(vector)
    }
}

pub fn validate_feature_vector_v3(features: &Vector) -> Result<(), FeatureError> {
    if features.len() != DCG_FEATURE_V3_COUNT {
        return Err(FeatureError::FeatureCountMismatch {
            expected: DCG_FEATURE_V3_COUNT,
            actual: features.len(),
        });
    }
    for (index, value) in features.iter().enumerate() {
        if !value.is_finite() {
            return Err(FeatureError::NonFiniteFeature {
                field: DCG_FEATURE_V3_NAMES[index],
                value: *value,
            });
        }
        if *value < 0.0 {
            return Err(FeatureError::InvalidFeatureValue {
                field: DCG_FEATURE_V3_NAMES[index],
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

fn root_property_names(schema: &Value) -> BTreeSet<String> {
    schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| properties.keys().cloned().collect())
        .unwrap_or_default()
}

fn type_bucket(schema: &Value) -> usize {
    match schema.get("type").and_then(Value::as_str) {
        Some("string") => 0,
        Some("number") | Some("integer") => 1,
        Some("boolean") => 2,
        Some("array") => 3,
        Some("object") => 4,
        _ => 5,
    }
}

fn root_additional_properties_tightened(base: &Value, candidate: &Value) -> bool {
    base.get("additionalProperties") != Some(&Value::Bool(false))
        && candidate.get("additionalProperties") == Some(&Value::Bool(false))
}

fn root_keyword_changed(base: &Value, candidate: &Value, keyword: &str) -> bool {
    base.get(keyword) != candidate.get(keyword) && candidate.get(keyword).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_added_field_type_and_root_restriction() {
        let base = r#"{"type":"object","properties":{"id":{"type":"string"}}}"#;
        let candidate = r#"{"type":"object","additionalProperties":false,"properties":{"id":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}}}}"#;
        let features = SchemaChangeFeatureV3Extractor::new()
            .extract(base, candidate, "baseline")
            .unwrap();
        assert_eq!(features.len(), DCG_FEATURE_V3_COUNT);
        assert_eq!(features[31], 1.0);
        assert_eq!(features[34], 1.0);
    }
}
