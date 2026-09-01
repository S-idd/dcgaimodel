use crate::linalg::Vector;

/// Stable identifier for the ordered DCG feature schema.
pub const DCG_FEATURE_VERSION: &str = "dcg-features-v1";
/// Number of values emitted by the DCG feature extractor.
pub const DCG_FEATURE_COUNT: usize = 8;
/// Stable feature names in the exact order consumed by models and artifacts.
pub const DCG_FEATURE_NAMES: [&str; DCG_FEATURE_COUNT] = [
    "field_count",
    "fields_added",
    "fields_removed",
    "type_changes",
    "compatibility_score",
    "semantic_version_ordinal",
    "breaking_change_count",
    "dependent_consumer_count",
];

/// Returns the canonical schema version and stable ordered feature names.
///
/// Ordering is part of model compatibility: values from a different schema
/// must never be supplied to a persisted `dcg-features-v1` model.
pub const fn feature_schema() -> (&'static str, &'static [&'static str; DCG_FEATURE_COUNT]) {
    (DCG_FEATURE_VERSION, &DCG_FEATURE_NAMES)
}

/// The compatibility assessment associated with a schema change.
///
/// Feature encoding is deterministic: `Compatible` is `0.0`, `Unknown` is
/// `0.5`, and `Incompatible` is `1.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityStatus {
    /// Existing consumers can use the changed schema.
    Compatible,
    /// Compatibility could not be established.
    Unknown,
    /// Existing consumers cannot safely use the changed schema.
    Incompatible,
}

impl CompatibilityStatus {
    /// Returns the documented numeric representation of this status.
    pub fn as_feature_value(self) -> f64 {
        match self {
            Self::Compatible => 0.0,
            Self::Unknown => 0.5,
            Self::Incompatible => 1.0,
        }
    }
}

/// A semantic schema version.
///
/// Its numeric feature is `major * 1_000_000 + minor * 1_000 + patch`; this
/// preserves the usual ordering for components below 1,000 while avoiding a
/// lossy string hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticVersion {
    /// Backward-incompatible release number.
    pub major: u32,
    /// Backward-compatible feature release number.
    pub minor: u32,
    /// Backward-compatible bug-fix release number.
    pub patch: u32,
}

impl SemanticVersion {
    /// Creates a semantic schema version.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns the documented numeric representation of this version.
    pub fn as_feature_value(self) -> f64 {
        self.major as f64 * 1_000_000.0 + self.minor as f64 * 1_000.0 + self.patch as f64
    }
}

/// Contract metadata with a direct, meaningful numerical interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ContractMetadata {
    /// Number of known downstream consumers affected by this contract.
    pub dependent_consumers: usize,
}

/// Source information about one changed data contract or schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractChange {
    /// Human-readable contract identifier. It is validated but not encoded.
    pub contract_name: String,
    /// Number of fields after the change.
    pub number_of_fields: usize,
    /// Number of fields introduced by the change.
    pub fields_added: usize,
    /// Number of fields removed by the change.
    pub fields_removed: usize,
    /// Number of fields whose data type changed.
    pub data_type_changes: usize,
    /// Compatibility assessment for the change.
    pub compatibility_status: CompatibilityStatus,
    /// Version assigned to the resulting schema.
    pub schema_version: SemanticVersion,
    /// Number of explicitly identified breaking changes.
    pub breaking_changes: usize,
    /// Relevant metadata with a meaningful risk interpretation.
    pub metadata: ContractMetadata,
}

/// Numeric DCG features accepted by the generic neural-network engine.
///
/// `to_vector` always emits these values in this order:
/// field count, fields added, fields removed, data-type changes,
/// compatibility encoding, schema-version encoding, breaking-change count,
/// and dependent-consumer count.
#[derive(Debug, Clone, PartialEq)]
pub struct ContractFeatures {
    /// Total number of fields after the change.
    pub number_of_fields: f64,
    /// Number of added fields.
    pub fields_added: f64,
    /// Number of removed fields.
    pub fields_removed: f64,
    /// Number of fields with data-type changes.
    pub data_type_changes: f64,
    /// Compatibility encoding: compatible `0.0`, unknown `0.5`, incompatible `1.0`.
    pub compatibility_status: f64,
    /// Semantic version encoded as `major * 1_000_000 + minor * 1_000 + patch`.
    pub schema_version: f64,
    /// Number of identified breaking changes.
    pub breaking_changes: f64,
    /// Number of known downstream contract consumers.
    pub dependent_consumers: f64,
}

impl ContractFeatures {
    /// Returns the feature vector in the stable, documented feature order.
    pub fn to_vector(&self) -> Vector {
        Vector::new(vec![
            self.number_of_fields,
            self.fields_added,
            self.fields_removed,
            self.data_type_changes,
            self.compatibility_status,
            self.schema_version,
            self.breaking_changes,
            self.dependent_consumers,
        ])
    }

    /// Returns the number of values emitted by [`Self::to_vector`].
    pub const fn feature_count() -> usize {
        DCG_FEATURE_COUNT
    }

    /// Returns the version of the schema emitted by [`Self::to_vector`].
    pub const fn feature_version() -> &'static str {
        DCG_FEATURE_VERSION
    }
}
