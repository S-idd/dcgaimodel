//! Policy configuration context for feature extraction.
//!
//! These values come directly from the approved policy-pack JSON. They are not
//! oracle outputs and do not inspect the generated compatibility label. The
//! resolution semantics intentionally mirror the pinned JAR's
//! `PolicyPackConfig`: every pack starts from the executable's eight-rule
//! baseline and then applies only that pack's explicit overrides.

use super::FeatureError;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Rule IDs and order used by the pinned JAR's `RuleId` enum.
pub const POLICY_RULE_IDS: [&str; 8] = [
    "FIELD_REMOVED",
    "FIELD_TYPE_CHANGED",
    "REQUIRED_FIELD_ADDED",
    "ENUM_VALUE_REMOVED",
    "ENUM_VALUE_ADDED",
    "CONSTRAINT_TIGHTENED",
    "CONDITIONAL_RESTRICTION_ADDED",
    "SCHEMA_RESTRICTION_ADDED",
];

pub const POLICY_RULE_COUNT: usize = POLICY_RULE_IDS.len();
pub const POLICY_ACTION_COUNT: usize = 3;
pub const POLICY_RULE_ACTION_FEATURE_COUNT: usize = POLICY_RULE_COUNT * POLICY_ACTION_COUNT;

const ENUM_VALUE_ADDED_INDEX: usize = 4;

/// Non-label policy configuration relevant to a schema transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PolicyFeatureContext {
    rule_actions: [PolicyRuleAction; POLICY_RULE_COUNT],
}

impl PolicyFeatureContext {
    /// One-hot policy action in stable IGNORE, WARNING, BREAKING order for
    /// each [`POLICY_RULE_IDS`] entry.
    pub fn rule_action_features(self) -> [f64; POLICY_RULE_ACTION_FEATURE_COUNT] {
        let mut values = [0.0; POLICY_RULE_ACTION_FEATURE_COUNT];
        for (rule_index, action) in self.rule_actions.iter().enumerate() {
            values[rule_index * POLICY_ACTION_COUNT + action.feature_index()] = 1.0;
        }
        values
    }

    /// Returns counts of the resolved actions for rules that are structurally
    /// active in one schema transition, in IGNORE/WARNING/BREAKING order.
    ///
    /// The caller supplies only schema-derived rule activity. This method does
    /// not inspect an oracle outcome or a training label. Collapsing active
    /// rule actions into this shared three-value representation lets a model
    /// learn the common meaning of an action from more than one rule family.
    pub fn active_rule_action_features(
        self,
        active_rules: &[bool; POLICY_RULE_COUNT],
    ) -> [f64; POLICY_ACTION_COUNT] {
        let mut values = [0.0; POLICY_ACTION_COUNT];
        for (active, action) in active_rules.iter().zip(self.rule_actions) {
            if *active {
                values[action.feature_index()] += 1.0;
            }
        }
        values
    }

    /// V4 compatibility helper for its historical enum-only policy context.
    pub const fn enum_value_added_features(self) -> [f64; 3] {
        self.rule_actions[ENUM_VALUE_ADDED_INDEX].one_hot()
    }
}

/// Resolved contexts for the approved policy packs in one immutable JSON file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovedPolicyContexts {
    contexts: BTreeMap<String, PolicyFeatureContext>,
}

impl ApprovedPolicyContexts {
    /// Reads and resolves each requested policy pack from the approved JSON.
    pub fn load(path: &Path, policy_packs: &[String]) -> Result<Self, FeatureError> {
        let document = fs::read_to_string(path).map_err(|error| FeatureError::PolicyContext {
            message: error.to_string(),
        })?;
        Self::from_json(&document, policy_packs)
    }

    /// Resolves each requested policy pack from a policy-pack JSON document.
    ///
    /// The resolution exactly matches the pinned Java implementation: an
    /// unknown request falls back to the configured default (or baseline), but
    /// declared packs do not inherit that default pack's overrides.
    pub fn from_json(document: &str, policy_packs: &[String]) -> Result<Self, FeatureError> {
        let root = serde_json::from_str::<Value>(document).map_err(|error| {
            FeatureError::PolicyContext {
                message: error.to_string(),
            }
        })?;
        let packs = root
            .get("packs")
            .and_then(Value::as_object)
            .ok_or_else(|| FeatureError::PolicyContext {
                message: "missing object packs".to_owned(),
            })?;
        let normalized_packs = normalized_packs(packs)?;
        let default_pack = root
            .get("defaultPack")
            .and_then(Value::as_str)
            .and_then(normalize_pack_name)
            .filter(|name| normalized_packs.contains_key(name))
            .unwrap_or_else(|| "baseline".to_owned());
        let mut contexts = BTreeMap::new();
        for policy_pack in policy_packs {
            let requested =
                normalize_pack_name(policy_pack).unwrap_or_else(|| default_pack.clone());
            let definition = normalized_packs
                .get(&requested)
                .or_else(|| normalized_packs.get(&default_pack));
            contexts.insert(
                policy_pack.clone(),
                PolicyFeatureContext {
                    rule_actions: resolve_rule_actions(definition.copied())?,
                },
            );
        }
        Ok(Self { contexts })
    }

    /// Returns the immutable resolved context for one requested pack.
    pub fn get(&self, policy_pack: &str) -> Result<PolicyFeatureContext, FeatureError> {
        self.contexts
            .get(policy_pack)
            .copied()
            .ok_or_else(|| FeatureError::PolicyContext {
                message: format!("policy pack `{policy_pack}` was not resolved"),
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PolicyRuleAction {
    Ignore,
    Warning,
    Breaking,
}

impl PolicyRuleAction {
    const fn feature_index(self) -> usize {
        match self {
            Self::Ignore => 0,
            Self::Warning => 1,
            Self::Breaking => 2,
        }
    }

    const fn one_hot(self) -> [f64; 3] {
        match self {
            Self::Ignore => [1.0, 0.0, 0.0],
            Self::Warning => [0.0, 1.0, 0.0],
            Self::Breaking => [0.0, 0.0, 1.0],
        }
    }

    fn parse(value: &str, rule: &str) -> Result<Self, FeatureError> {
        match value {
            "IGNORE" => Ok(Self::Ignore),
            "WARNING" => Ok(Self::Warning),
            "BREAKING" => Ok(Self::Breaking),
            _ => Err(FeatureError::PolicyContext {
                message: format!(
                    "{rule} action must be IGNORE, WARNING, or BREAKING; got `{value}`"
                ),
            }),
        }
    }
}

const BASELINE_RULE_ACTIONS: [PolicyRuleAction; POLICY_RULE_COUNT] = [
    PolicyRuleAction::Breaking,
    PolicyRuleAction::Breaking,
    PolicyRuleAction::Breaking,
    PolicyRuleAction::Breaking,
    PolicyRuleAction::Warning,
    PolicyRuleAction::Breaking,
    PolicyRuleAction::Breaking,
    PolicyRuleAction::Breaking,
];

fn normalized_packs(
    packs: &serde_json::Map<String, Value>,
) -> Result<BTreeMap<String, &Value>, FeatureError> {
    let mut normalized = BTreeMap::new();
    for (name, definition) in packs {
        let Some(name) = normalize_pack_name(name) else {
            continue;
        };
        if normalized
            .insert(name.to_ascii_lowercase(), definition)
            .is_some()
        {
            return Err(FeatureError::PolicyContext {
                message: format!("policy pack name `{name}` is duplicated after normalization"),
            });
        }
    }
    Ok(normalized)
}

fn normalize_pack_name(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_ascii_lowercase())
}

fn resolve_rule_actions(
    definition: Option<&Value>,
) -> Result<[PolicyRuleAction; POLICY_RULE_COUNT], FeatureError> {
    let mut actions = BASELINE_RULE_ACTIONS;
    let Some(definition) = definition else {
        return Ok(actions);
    };
    if definition.is_null() {
        return Ok(actions);
    }
    let definition = definition
        .as_object()
        .ok_or_else(|| FeatureError::PolicyContext {
            message: "policy pack definition must be an object or null".to_owned(),
        })?;
    let Some(rules) = definition.get("rules") else {
        return Ok(actions);
    };
    if rules.is_null() {
        return Ok(actions);
    }
    let rules = rules
        .as_object()
        .ok_or_else(|| FeatureError::PolicyContext {
            message: "policy pack rules must be an object or null".to_owned(),
        })?;
    for (raw_rule, value) in rules {
        let Some(normalized_rule) = normalize_rule_key(raw_rule) else {
            continue;
        };
        let rule_index = POLICY_RULE_IDS
            .iter()
            .position(|rule| *rule == normalized_rule)
            .ok_or_else(|| FeatureError::PolicyContext {
                message: format!("unknown policy rule id `{raw_rule}`"),
            })?;
        // The pinned JAR turns an explicit JSON null into BREAKING.
        actions[rule_index] = match value.as_str() {
            Some(action) => PolicyRuleAction::parse(action, &normalized_rule)?,
            None if value.is_null() => PolicyRuleAction::Breaking,
            None => {
                return Err(FeatureError::PolicyContext {
                    message: format!("{normalized_rule} action must be a string or null"),
                });
            }
        };
    }
    Ok(actions)
}

fn normalize_rule_key(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_uppercase().replace('-', "_"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_all_effective_rule_actions_without_using_labels() {
        let contexts = ApprovedPolicyContexts::from_json(
            r#"{
              "defaultPack":"baseline",
              "packs": {
                "baseline":{"rules":{"ENUM_VALUE_ADDED":"WARNING"}},
                "strict":{"rules":{"ENUM_VALUE_ADDED":"BREAKING","CONSTRAINT_TIGHTENED":"IGNORE"}},
                "inherits":{"rules":{}}
              }
            }"#,
            &[
                "baseline".to_owned(),
                "strict".to_owned(),
                "inherits".to_owned(),
            ],
        )
        .unwrap();
        assert_eq!(
            contexts.get("baseline").unwrap().rule_action_features(),
            [
                0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
            ]
        );
        assert_eq!(
            contexts.get("strict").unwrap().enum_value_added_features(),
            [0.0, 0.0, 1.0]
        );
        assert_eq!(
            &contexts.get("strict").unwrap().rule_action_features()[15..18],
            &[1.0, 0.0, 0.0]
        );
        // The JAR starts every declared pack from executable baseline rules;
        // it does not inherit custom overrides from the default pack.
        assert_eq!(
            contexts
                .get("inherits")
                .unwrap()
                .enum_value_added_features(),
            [0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn normalizes_rule_keys_and_rejects_unknown_actions() {
        let normalized = ApprovedPolicyContexts::from_json(
            r#"{"packs":{"baseline":{"rules":{"constraint-tightened":"WARNING"}}}}"#,
            &["baseline".to_owned()],
        )
        .unwrap();
        assert_eq!(
            &normalized.get("baseline").unwrap().rule_action_features()[15..18],
            &[0.0, 1.0, 0.0]
        );
        assert!(
            ApprovedPolicyContexts::from_json(
                r#"{"packs":{"baseline":{"rules":{"FIELD_REMOVED":"PASS"}}}}"#,
                &["baseline".to_owned()],
            )
            .is_err()
        );
    }

    #[test]
    fn resolves_the_approved_v5_compositional_profiles() {
        let document = include_str!("../../tests/fixtures/policy-packs-v5-compositional.json");
        let contexts = ApprovedPolicyContexts::from_json(
            document,
            &[
                "composition-enum-ignore-constraint-warning".to_owned(),
                "composition-enum-warning-constraint-ignore".to_owned(),
                "composition-enum-breaking-constraint-breaking".to_owned(),
            ],
        )
        .unwrap();
        // Enum-value-added occupies action slots 12..15; constraint-tightened
        // occupies slots 15..18. The action order is IGNORE/WARNING/BREAKING.
        assert_eq!(
            &contexts
                .get("composition-enum-ignore-constraint-warning")
                .unwrap()
                .rule_action_features()[12..18],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            &contexts
                .get("composition-enum-warning-constraint-ignore")
                .unwrap()
                .rule_action_features()[12..18],
            &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            &contexts
                .get("composition-enum-breaking-constraint-breaking")
                .unwrap()
                .rule_action_features()[12..18],
            &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
        );
    }
}
