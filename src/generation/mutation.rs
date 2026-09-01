//! Structural JSON Schema mutations proposed to the external oracle.
//!
//! These mutations deliberately do not assign labels. A proposed mutation can
//! be unsupported or have a policy-specific result; only the pinned oracle's
//! exit code decides whether it becomes a training record.

use serde_json::{Map, Value, json};

/// A structural change shape that can be proposed to the oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutationKind {
    /// Remove an existing root-level property.
    FieldRemoved,
    /// Change a root-level property's scalar `type`.
    FieldTypeChanged,
    /// Mark an existing root-level property required.
    RequiredFieldAdded,
    /// Remove one value from an existing enum.
    EnumValueRemoved,
    /// Add a value to an existing enum.
    EnumValueAdded,
    /// Add or strengthen a tracked property constraint.
    ConstraintTightened,
    /// Add a root-level JSON Schema `if`/`then` restriction.
    ConditionalRestrictionAdded,
    /// Add a root-level schema restriction.
    SchemaRestrictionAdded,
    /// Add an optional property, producing a useful compatible control case.
    OptionalFieldAdded,
}

impl MutationKind {
    /// Parses the portable mutation identifier accepted by CLI filters.
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "field_removed" => Self::FieldRemoved,
            "field_type_changed" => Self::FieldTypeChanged,
            "required_field_added" => Self::RequiredFieldAdded,
            "enum_value_removed" => Self::EnumValueRemoved,
            "enum_value_added" => Self::EnumValueAdded,
            "constraint_tightened" => Self::ConstraintTightened,
            "conditional_restriction_added" => Self::ConditionalRestrictionAdded,
            "schema_restriction_added" => Self::SchemaRestrictionAdded,
            "optional_field_added" => Self::OptionalFieldAdded,
            _ => return None,
        })
    }

    /// Stable, portable mutation identifier for audit metadata.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::FieldRemoved => "field_removed",
            Self::FieldTypeChanged => "field_type_changed",
            Self::RequiredFieldAdded => "required_field_added",
            Self::EnumValueRemoved => "enum_value_removed",
            Self::EnumValueAdded => "enum_value_added",
            Self::ConstraintTightened => "constraint_tightened",
            Self::ConditionalRestrictionAdded => "conditional_restriction_added",
            Self::SchemaRestrictionAdded => "schema_restriction_added",
            Self::OptionalFieldAdded => "optional_field_added",
        }
    }
}

/// A candidate pair whose compatibility must be determined by the oracle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MutationCandidate {
    /// The structural operation proposed, retained only for provenance.
    pub kind: MutationKind,
    /// Deterministic structural form within a mutation family. This is
    /// provenance only; it is never a model feature or oracle label.
    pub variant: String,
    /// Candidate JSON Schema text for the newer contract version.
    pub candidate_schema: String,
}

/// Creates all applicable single-change candidates for one JSON Schema seed.
///
/// Input must be a root JSON object with a `properties` object. Unsupported
/// schemas simply yield fewer candidates and are later rejected by the oracle
/// if they use JSON Schema constructs outside the pinned CLI's support.
pub fn candidates_for_schema(schema: &str) -> Result<Vec<MutationCandidate>, serde_json::Error> {
    let base = serde_json::from_str::<Value>(schema)?;
    Ok(candidates_for_value(&base))
}

/// Creates candidate mutations from a parsed schema. Exposed for unit tests
/// and callers that already validated JSON input.
pub fn candidates_for_value(base: &Value) -> Vec<MutationCandidate> {
    let Some(properties) = root_properties(base) else {
        return Vec::new();
    };
    // Canonicalise root-property traversal so candidate IDs do not depend on
    // the JSON object's incidental insertion order.
    let mut property_names = properties.keys().cloned().collect::<Vec<_>>();
    property_names.sort();
    if property_names.is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::new();
    let first_property = &property_names[0];

    if let Some(mut candidate) = root_object_copy(base) {
        candidate
            .get_mut("properties")
            .and_then(Value::as_object_mut)
            .expect("root properties were checked")
            .remove(first_property);
        if let Some(required) = candidate.get_mut("required").and_then(Value::as_array_mut) {
            required.retain(|name| name.as_str() != Some(first_property));
        }
        push(
            &mut candidates,
            MutationKind::FieldRemoved,
            "remove-first-root-property",
            candidate,
        );
    }

    for name in &property_names {
        let Some(original) = properties.get(name).and_then(Value::as_object) else {
            continue;
        };
        let Some(old_type) = original.get("type").and_then(Value::as_str) else {
            continue;
        };
        let replacement = replacement_type(old_type);
        if let Some(mut candidate) = root_object_copy(base)
            && let Some(property) = candidate
                .get_mut("properties")
                .and_then(Value::as_object_mut)
                .and_then(|values| values.get_mut(name))
                .and_then(Value::as_object_mut)
        {
            property.insert("type".to_owned(), Value::String(replacement.to_owned()));
            push(
                &mut candidates,
                MutationKind::FieldTypeChanged,
                "change-first-supported-root-type",
                candidate,
            );
            break;
        }
    }

    let required = base
        .get("required")
        .and_then(Value::as_array)
        .map(|values| values.iter().filter_map(Value::as_str).collect::<Vec<_>>())
        .unwrap_or_default();
    if (base.get("required").is_none() || base.get("required").is_some_and(Value::is_array))
        && let Some(name) = property_names
            .iter()
            .find(|name| !required.iter().any(|required_name| required_name == name))
        && let Some(mut candidate) = root_object_copy(base)
    {
        let root = candidate.as_object_mut().expect("root object was checked");
        let values = root
            .entry("required")
            .or_insert_with(|| Value::Array(Vec::new()))
            .as_array_mut()
            .expect("required is checked as an array before mutation");
        values.push(Value::String(name.to_owned()));
        push(
            &mut candidates,
            MutationKind::RequiredFieldAdded,
            "make-first-optional-root-property-required",
            candidate,
        );
    }

    for name in &property_names {
        let Some(values) = properties
            .get(name)
            .and_then(Value::as_object)
            .and_then(|property| property.get("enum"))
            .and_then(Value::as_array)
        else {
            continue;
        };
        if values.len() >= 2
            && let Some(mut candidate) = root_object_copy(base)
        {
            candidate
                .get_mut("properties")
                .and_then(Value::as_object_mut)
                .and_then(|properties| properties.get_mut(name))
                .and_then(Value::as_object_mut)
                .and_then(|property| property.get_mut("enum"))
                .and_then(Value::as_array_mut)
                .expect("enum was checked")
                .pop();
            push(
                &mut candidates,
                MutationKind::EnumValueRemoved,
                "remove-last-enum-value",
                candidate,
            );
        }
        break;
    }

    // The prior generator always appended one fixed string to the first enum.
    // Generate bounded, type-preserving variants over several independent
    // enum-bearing fields instead. The JAR still labels every transition.
    for (ordinal, name) in property_names.iter().enumerate().take(4) {
        let Some(values) = properties
            .get(name)
            .and_then(Value::as_object)
            .and_then(|property| property.get("enum"))
            .and_then(Value::as_array)
        else {
            continue;
        };
        let Some(value) = novel_enum_value(values, ordinal) else {
            continue;
        };
        if let Some(mut candidate) = root_object_copy(base) {
            candidate
                .get_mut("properties")
                .and_then(Value::as_object_mut)
                .and_then(|properties| properties.get_mut(name))
                .and_then(Value::as_object_mut)
                .and_then(|property| property.get_mut("enum"))
                .and_then(Value::as_array_mut)
                .expect("enum was checked")
                .push(value);
            push(
                &mut candidates,
                MutationKind::EnumValueAdded,
                &format!("add-type-preserving-enum-value-{ordinal}"),
                candidate,
            );
        }
    }

    // Rich constraint families are deliberately proposed as distinct schema
    // transitions. They do not carry a Rust label: the pinned JAR decides
    // whether a candidate is SAFE, WARNING, BREAKING, or rejected.
    //
    // The candidate values avoid contradictory lower/upper bounds, which
    // keeps the generated JSON Schemas valid rather than relying on the oracle
    // to reject malformed examples. Every variant name includes a stable root
    // property ordinal, so each retained record has a unique portable ID.
    for (ordinal, name) in property_names.iter().enumerate().take(4) {
        let Some(property) = properties.get(name).and_then(Value::as_object) else {
            continue;
        };
        match property.get("type").and_then(Value::as_str) {
            Some("integer") | Some("number") => {
                if let Some(minimum) = new_numeric_minimum(property) {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-numeric-minimum-{ordinal}"),
                        vec![("minimum", minimum)],
                    );
                }
                if let Some(maximum) = new_numeric_maximum(property) {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-numeric-maximum-{ordinal}"),
                        vec![("maximum", maximum)],
                    );
                }
                if !property.contains_key("multipleOf") {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-numeric-multiple-of-{ordinal}"),
                        vec![("multipleOf", json!(2))],
                    );
                }
                if let Some(minimum) = new_numeric_minimum(property)
                    && !property.contains_key("multipleOf")
                {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-numeric-minimum-multiple-of-{ordinal}"),
                        vec![("minimum", minimum), ("multipleOf", json!(2))],
                    );
                }
            }
            Some("string") => {
                if let Some(minimum) = new_count_lower_bound(property, "minLength", "maxLength") {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-string-min-length-{ordinal}"),
                        vec![("minLength", minimum)],
                    );
                }
                if let Some(maximum) = new_count_upper_bound(property, "minLength", "maxLength") {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-string-max-length-{ordinal}"),
                        vec![("maxLength", maximum)],
                    );
                }
                if !property.contains_key("pattern") {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-string-pattern-{ordinal}"),
                        vec![("pattern", json!("^[A-Za-z][A-Za-z0-9_-]{0,63}$"))],
                    );
                }
                if let Some(minimum) = new_count_lower_bound(property, "minLength", "maxLength")
                    && !property.contains_key("pattern")
                {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-string-min-length-pattern-{ordinal}"),
                        vec![
                            ("minLength", minimum),
                            ("pattern", json!("^[A-Za-z][A-Za-z0-9_-]{0,63}$")),
                        ],
                    );
                }
            }
            Some("array") => {
                if let Some(minimum) = new_count_lower_bound(property, "minItems", "maxItems") {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-array-min-items-{ordinal}"),
                        vec![("minItems", minimum)],
                    );
                }
                if let Some(maximum) = new_count_upper_bound(property, "minItems", "maxItems") {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-array-max-items-{ordinal}"),
                        vec![("maxItems", maximum)],
                    );
                }
                if property.get("uniqueItems") != Some(&Value::Bool(true)) {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-array-unique-items-{ordinal}"),
                        vec![("uniqueItems", Value::Bool(true))],
                    );
                }
                if let Some(minimum) = new_count_lower_bound(property, "minItems", "maxItems")
                    && property.get("uniqueItems") != Some(&Value::Bool(true))
                {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-array-min-items-unique-items-{ordinal}"),
                        vec![("minItems", minimum), ("uniqueItems", Value::Bool(true))],
                    );
                }
            }
            Some("object") => {
                if let Some(minimum) =
                    new_count_lower_bound(property, "minProperties", "maxProperties")
                {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-object-min-properties-{ordinal}"),
                        vec![("minProperties", minimum)],
                    );
                }
                if let Some(maximum) =
                    new_count_upper_bound(property, "minProperties", "maxProperties")
                {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-object-max-properties-{ordinal}"),
                        vec![("maxProperties", maximum)],
                    );
                }
                if let (Some(minimum), Some(maximum)) = (
                    new_count_lower_bound(property, "minProperties", "maxProperties"),
                    new_count_upper_bound(property, "minProperties", "maxProperties"),
                ) {
                    push_constraint_candidate(
                        &mut candidates,
                        base,
                        name,
                        format!("tighten-object-min-max-properties-{ordinal}"),
                        vec![("minProperties", minimum), ("maxProperties", maximum)],
                    );
                }
            }
            _ => {}
        }
    }

    if !base.get("if").is_some_and(Value::is_object)
        && !base.get("then").is_some_and(Value::is_object)
        && let Some(mut candidate) = root_object_copy(base)
    {
        let mut condition_properties = Map::new();
        condition_properties.insert(
            first_property.to_owned(),
            json!({"const": "__dcg_generated__"}),
        );
        let root = candidate.as_object_mut().expect("root object was checked");
        root.insert("if".to_owned(), json!({"properties": condition_properties}));
        root.insert(
            "then".to_owned(),
            json!({"required": [first_property.to_owned()]}),
        );
        push(
            &mut candidates,
            MutationKind::ConditionalRestrictionAdded,
            "root-if-then-required",
            candidate,
        );
    }

    if base.get("additionalProperties") != Some(&Value::Bool(false))
        && let Some(mut candidate) = root_object_copy(base)
    {
        candidate
            .as_object_mut()
            .expect("root object was checked")
            .insert("additionalProperties".to_owned(), Value::Bool(false));
        push(
            &mut candidates,
            MutationKind::SchemaRestrictionAdded,
            "root-additional-properties-false",
            candidate,
        );
    }

    if base.get("propertyNames").is_none()
        && let Some(mut candidate) = root_object_copy(base)
    {
        candidate
            .as_object_mut()
            .expect("root object was checked")
            .insert(
                "propertyNames".to_owned(),
                json!({"pattern": "^[A-Za-z_][A-Za-z0-9_]*$"}),
            );
        push(
            &mut candidates,
            MutationKind::SchemaRestrictionAdded,
            "root-property-names-pattern",
            candidate,
        );
    }
    if property_names.len() >= 2
        && base.get("dependentRequired").is_none()
        && let Some(mut candidate) = root_object_copy(base)
    {
        candidate
            .as_object_mut()
            .expect("root object was checked")
            .insert(
                "dependentRequired".to_owned(),
                json!({property_names[0].clone(): [property_names[1].clone()]}),
            );
        push(
            &mut candidates,
            MutationKind::SchemaRestrictionAdded,
            "root-dependent-required",
            candidate,
        );
    }

    for (variant, schema) in [
        ("string", json!({"type": "string"})),
        ("integer", json!({"type": "integer"})),
        ("boolean", json!({"type": "boolean"})),
        (
            "array",
            json!({"type": "array", "items": {"type": "string"}}),
        ),
    ] {
        if let Some(mut candidate) = root_object_copy(base) {
            let mut added_name = format!("dcg_generated_optional_{variant}");
            while properties.contains_key(&added_name) {
                added_name.push('_');
            }
            candidate
                .get_mut("properties")
                .and_then(Value::as_object_mut)
                .expect("root properties were checked")
                .insert(added_name, schema);
            push(
                &mut candidates,
                MutationKind::OptionalFieldAdded,
                &format!("optional-{variant}-field"),
                candidate,
            );
        }
    }

    candidates
}

fn root_properties(base: &Value) -> Option<&Map<String, Value>> {
    base.as_object()?.get("properties")?.as_object()
}

fn root_object_copy(base: &Value) -> Option<Value> {
    base.as_object().map(|_| base.clone())
}

fn push(
    candidates: &mut Vec<MutationCandidate>,
    kind: MutationKind,
    variant: &str,
    candidate: Value,
) {
    if let Ok(candidate_schema) = serde_json::to_string(&candidate) {
        candidates.push(MutationCandidate {
            kind,
            variant: variant.to_owned(),
            candidate_schema,
        });
    }
}

fn push_constraint_candidate(
    candidates: &mut Vec<MutationCandidate>,
    base: &Value,
    property_name: &str,
    variant: String,
    constraints: Vec<(&str, Value)>,
) {
    let Some(mut candidate) = root_object_copy(base) else {
        return;
    };
    let Some(property) = candidate
        .get_mut("properties")
        .and_then(Value::as_object_mut)
        .and_then(|properties| properties.get_mut(property_name))
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    for (keyword, value) in constraints {
        property.insert(keyword.to_owned(), value);
    }
    push(
        candidates,
        MutationKind::ConstraintTightened,
        &variant,
        candidate,
    );
}

fn new_numeric_minimum(property: &Map<String, Value>) -> Option<Value> {
    if property.contains_key("minimum") {
        return None;
    }
    let maximum = property.get("maximum").and_then(Value::as_f64);
    let value = maximum.map_or(0.0, |maximum| maximum - 1.0);
    serde_json::Number::from_f64(value).map(Value::Number)
}

fn new_numeric_maximum(property: &Map<String, Value>) -> Option<Value> {
    if property.contains_key("maximum") {
        return None;
    }
    let minimum = property.get("minimum").and_then(Value::as_f64);
    let value = minimum.map_or(100.0, |minimum| minimum + 1.0);
    serde_json::Number::from_f64(value).map(Value::Number)
}

fn new_count_lower_bound(
    property: &Map<String, Value>,
    lower_keyword: &str,
    upper_keyword: &str,
) -> Option<Value> {
    if property.contains_key(lower_keyword)
        || property
            .get(upper_keyword)
            .and_then(Value::as_u64)
            .is_some_and(|maximum| maximum == 0)
    {
        return None;
    }
    Some(json!(1))
}

fn new_count_upper_bound(
    property: &Map<String, Value>,
    lower_keyword: &str,
    upper_keyword: &str,
) -> Option<Value> {
    if property.contains_key(upper_keyword) {
        return None;
    }
    let minimum = property
        .get(lower_keyword)
        .and_then(Value::as_u64)
        .unwrap_or(0);
    minimum.checked_add(8).map(|maximum| json!(maximum))
}

fn novel_enum_value(values: &[Value], ordinal: usize) -> Option<Value> {
    let candidate = match values.first()? {
        Value::String(_) => Value::String(format!("__dcg_generated_enum_value_{ordinal}__")),
        Value::Number(_) => json!(1_000_000_u64 + ordinal as u64),
        Value::Bool(_) => {
            let contains_true = values.iter().any(|value| value == &Value::Bool(true));
            let contains_false = values.iter().any(|value| value == &Value::Bool(false));
            match (contains_true, contains_false) {
                (true, false) => Value::Bool(false),
                (false, true) => Value::Bool(true),
                _ => return None,
            }
        }
        _ => return None,
    };
    (!values.contains(&candidate)).then_some(candidate)
}

fn replacement_type(previous: &str) -> &'static str {
    match previous {
        "string" => "integer",
        "integer" | "number" => "string",
        "boolean" => "string",
        "array" => "object",
        "object" => "array",
        _ => "string",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_portable_mutation_identifiers() {
        assert_eq!(
            MutationKind::parse("enum_value_removed"),
            Some(MutationKind::EnumValueRemoved)
        );
        assert_eq!(MutationKind::parse("unknown"), None);
    }

    #[test]
    fn proposes_all_supported_rule_shapes_when_preconditions_exist() {
        let schema = r#"{
          "type": "object",
          "properties": {
            "kind": {"type": "string", "enum": ["a", "b"]},
            "count": {"type": "integer"}
          },
          "required": ["kind"]
        }"#;
        let kinds = candidates_for_schema(schema)
            .unwrap()
            .into_iter()
            .map(|candidate| candidate.kind)
            .collect::<Vec<_>>();
        for kind in [
            MutationKind::FieldRemoved,
            MutationKind::FieldTypeChanged,
            MutationKind::RequiredFieldAdded,
            MutationKind::EnumValueRemoved,
            MutationKind::EnumValueAdded,
            MutationKind::ConstraintTightened,
            MutationKind::ConditionalRestrictionAdded,
            MutationKind::SchemaRestrictionAdded,
            MutationKind::OptionalFieldAdded,
        ] {
            assert!(kinds.contains(&kind), "missing {}", kind.as_str());
        }
    }

    #[test]
    fn proposes_rich_independent_constraint_tightening_variants() {
        let schema = r#"{
          "type": "object",
          "properties": {
            "amount": {"type": "number"},
            "name": {"type": "string"},
            "payload": {"type": "object", "properties": {"kind": {"type": "string"}}},
            "tags": {"type": "array", "items": {"type": "string"}}
          }
        }"#;
        let candidates = candidates_for_schema(schema).unwrap();
        let variants = candidates
            .iter()
            .filter(|candidate| candidate.kind == MutationKind::ConstraintTightened)
            .map(|candidate| candidate.variant.as_str())
            .collect::<Vec<_>>();
        for expected in [
            "tighten-numeric-minimum-0",
            "tighten-numeric-minimum-multiple-of-0",
            "tighten-string-min-length-1",
            "tighten-string-pattern-1",
            "tighten-string-min-length-pattern-1",
            "tighten-object-min-properties-2",
            "tighten-object-min-max-properties-2",
            "tighten-array-min-items-3",
            "tighten-array-min-items-unique-items-3",
        ] {
            assert!(variants.contains(&expected), "missing {expected}");
        }

        let mixed = candidates
            .iter()
            .find(|candidate| candidate.variant == "tighten-string-min-length-pattern-1")
            .unwrap();
        let generated = serde_json::from_str::<Value>(&mixed.candidate_schema).unwrap();
        let name = generated
            .get("properties")
            .and_then(Value::as_object)
            .and_then(|properties| properties.get("name"))
            .and_then(Value::as_object)
            .unwrap();
        assert_eq!(name.get("minLength"), Some(&json!(1)));
        assert_eq!(
            name.get("pattern"),
            Some(&json!("^[A-Za-z][A-Za-z0-9_-]{0,63}$"))
        );
    }
}
