//! Portable prepared DCG dataset records and family-aware partitioning.

use super::{
    CompatibilityLabel, ContractLabels, DcgDatasetError, OracleCompatibilityMode,
    OracleInvariantPromotionManifest, TargetMode,
};
use crate::dataset::{Dataset, DatasetError, DatasetSplitConfig};
use crate::features::{
    ContractFeatureExtractor, DCG_FEATURE_V2_VERSION, DCG_FEATURE_V3_VERSION,
    DCG_FEATURE_V4_VERSION, DCG_FEATURE_V5_VERSION, DCG_FEATURE_V6_VERSION, DCG_FEATURE_VERSION,
    DcgFixture, FeatureError, FeatureExtractor, FixturePolicyPack,
    validate_feature_vector_for_version,
};
use crate::linalg::Vector;
use crate::prediction::PredictionKind;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

/// Version of the portable prepared-dataset JSON representation.
pub const PREPARED_DATASET_FORMAT_VERSION: &str = "dcg-prepared-dataset-v2";

/// Intended evaluation role for a complete source family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetRole {
    /// Available for deterministic train/validation/test splitting.
    Standard,
    /// Reserved whole-family challenge data; never enters normal splitting.
    Challenge,
}

impl DatasetRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Challenge => "challenge",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "standard" => Some(Self::Standard),
            "challenge" => Some(Self::Challenge),
            _ => None,
        }
    }
}

/// Oracle evidence retained for records made by the offline generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedRecordProvenance {
    /// SHA-256 of the exact executable that produced the compatibility label.
    pub oracle_jar_sha256: String,
    /// SHA-256 of the policy-pack file copied into the oracle staging directory.
    pub policy_packs_sha256: String,
    /// The oracle exit classification: `pass` or `breaking`.
    pub oracle_outcome: String,
    /// Structural mutation proposal; this is provenance, never a label.
    pub declared_mutation: String,
    /// Concrete structural form within that mutation family.
    pub mutation_variant: String,
    /// Label-free root-object context of the source pair. New generator
    /// artifacts record `open` or `closed`; legacy artifacts may omit it.
    pub root_object_profile: Option<String>,
    /// Captured CLI output from the labeling invocation.
    pub oracle_stdout: String,
    /// Compatibility direction passed to the executable oracle.
    pub compatibility_mode: String,
    /// Generator implementation version responsible for the record.
    pub generator_version: String,
    /// Deterministic seed used for source selection and split replay.
    pub generation_seed: u64,
    /// Explicit label authority; generated records must name the pinned oracle.
    pub label_source: String,
    /// Legacy generator-side rationale retained for backward-compatible reads.
    /// Benchmark exceptions now require a qualified promotion manifest.
    pub oracle_invariant_rationale: Option<String>,
    /// SHA-256 over canonical base/candidate JSON plus policy and direction.
    pub pair_fingerprint: String,
}

/// One portable, policy-labelled DCG record ready for feature-model training.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedDcgRecord {
    /// Stable record identifier; not a local filesystem path.
    pub record_id: String,
    /// Dataset/source provenance identifier.
    pub source: String,
    /// Stable schema/contract family fingerprint or identifier.
    pub family_id: String,
    /// All sibling variants in this group must remain in one partition.
    pub split_group_id: String,
    /// Normal or reserved challenge role, assigned consistently per family.
    pub dataset_role: DatasetRole,
    /// Logical DCG contract identifier.
    pub contract_id: String,
    /// Original schema version identifier.
    pub old_version: String,
    /// Resulting schema version identifier.
    pub new_version: String,
    /// Fully resolved policy pack identifier.
    pub policy_pack: String,
    /// Versioned canonical feature schema.
    pub feature_version: String,
    /// Validated raw DCG feature vector.
    pub features: Vector,
    /// Deterministic policy-derived targets for supported model tasks.
    pub labels: ContractLabels,
    /// Original SAFE/WARNING/BREAKING oracle label when this is generated data.
    pub compatibility_label: Option<CompatibilityLabel>,
    /// Evidence for generated records, absent for hand-maintained fixtures.
    pub generation: Option<GeneratedRecordProvenance>,
}

/// A portable prepared dataset with shared versioned metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedDcgDataset {
    dataset_version: String,
    /// Persisted corpus-design decision. `None` represents a legacy or
    /// otherwise unstamped artifact and is deliberately not treated as ready.
    benchmark_ready: Option<bool>,
    records: Vec<PreparedDcgRecord>,
}

/// Family-aware train/validation/test partitions of a prepared dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedDcgDatasetSplit {
    train: PreparedDcgDataset,
    validation: PreparedDcgDataset,
    test: PreparedDcgDataset,
}

/// Distribution and diversity facts used to gate a serious training run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetStatistics {
    /// Total portable records.
    pub total_records: usize,
    /// Distinct source/split families.
    pub independent_families: usize,
    /// Canonical SAFE record count.
    pub safe: usize,
    /// Canonical WARNING record count.
    pub warning: usize,
    /// Canonical BREAKING record count.
    pub breaking: usize,
    /// Distinct policy pack names.
    pub policy_packs: usize,
    /// Distinct generator mutation categories when evidence exists.
    pub mutation_categories: usize,
    /// Shared feature schema version.
    pub feature_version: String,
}

/// Explicit readiness result; this is not a claim of production adequacy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingReadinessReport {
    /// Dataset facts considered by the gate.
    pub statistics: DatasetStatistics,
    /// Requested target view.
    pub target_mode: TargetMode,
    /// Record counts after deterministic group splitting, when possible.
    pub partition_records: Option<(usize, usize, usize)>,
    /// Complete family counts after deterministic group splitting, when possible.
    pub partition_families: Option<(usize, usize, usize)>,
    /// Whether family leakage was detected by the split audit.
    pub leakage_detected: bool,
    /// Whether structural prerequisites have been met.
    pub ready_for_training: bool,
    /// Explicit reasons when the structural gate is not satisfied.
    pub reasons: Vec<String>,
}

/// One named, independently held-out challenge view over reserved families.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChallengeProtocol {
    /// Evaluate only challenge records labelled under one policy pack.
    HeldOutPolicy { policy_pack: String },
    /// Evaluate only challenge records proposed by one mutation family.
    HeldOutMutation { mutation: String },
    /// Evaluate counterfactual challenge pairs that retain the same mutation
    /// under at least two different policy packs.
    SameMutationDifferentPolicy,
}

impl ChallengeProtocol {
    /// Stable, human-readable identifier retained in reports and CLI output.
    pub fn as_str(&self) -> String {
        match self {
            Self::HeldOutPolicy { policy_pack } => format!("held-out-policy:{policy_pack}"),
            Self::HeldOutMutation { mutation } => format!("held-out-mutation:{mutation}"),
            Self::SameMutationDifferentPolicy => "same-mutation-different-policy".to_owned(),
        }
    }
}

/// Readiness facts for one challenge protocol. `meaningful` requires more than
/// one oracle class; a single-class challenge is retained for inspection but
/// must not be reported as a generalization benchmark.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChallengeProtocolReport {
    pub protocol: String,
    pub records: usize,
    pub families: usize,
    pub policies: usize,
    pub mutations: usize,
    pub labels: BTreeMap<String, usize>,
    pub meaningful: bool,
    pub reason: Option<String>,
}

/// A label-pure mutation that is explicitly documented as deterministic-oracle
/// invariant rather than silently exempted from shortcut checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleInvariantException {
    pub mutation: String,
    pub observed_policies: Vec<String>,
    pub observed_labels: BTreeMap<String, usize>,
    pub rationale: String,
}

/// Family-level coverage of optional-field additions by the source root's
/// label-free open/closed object profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionalFieldProfileCoverage {
    pub standard_families: usize,
    pub challenge_families: usize,
}

/// Complete shortcut and counterfactual-coverage audit computed from a saved
/// prepared dataset. This assesses corpus design, never model accuracy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneralizationReadinessReport {
    pub policy_by_label: BTreeMap<String, BTreeMap<String, usize>>,
    pub mutation_by_label: BTreeMap<String, BTreeMap<String, usize>>,
    pub policy_mutation_by_label: BTreeMap<String, BTreeMap<String, usize>>,
    /// Mutations observed with one oracle label across the available policies.
    /// These are recorded as observed oracle-invariant exceptions, not treated
    /// as fabricated counterfactual requirements.
    pub observed_oracle_invariant_mutations: Vec<String>,
    pub oracle_invariant_exceptions: Vec<OracleInvariantException>,
    /// Label-pure policy or non-exempt policy/mutation stratum correlations.
    pub shortcut_risks: Vec<String>,
    pub challenge_records: usize,
    pub standard_families: usize,
    pub challenge_families: usize,
    pub challenge_family_leakage: bool,
    pub challenge_pair_leakage: bool,
    pub family_mutation_pairs: usize,
    pub complete_policy_counterfactual_pairs: usize,
    /// Explicit root-profile coverage needed for an honest optional-field
    /// structural challenge. A V6 feature contains this fact, but model
    /// provenance must preserve it too so evaluation can hold it out.
    pub optional_field_profile_coverage: BTreeMap<String, OptionalFieldProfileCoverage>,
    pub challenge_protocols: Vec<ChallengeProtocolReport>,
    pub ready_for_generalization_benchmark: bool,
    pub reasons: Vec<String>,
}

/// Errors returned by prepared dataset validation, persistence, and splitting.
#[derive(Debug)]
pub enum PreparedDatasetError {
    /// Required portable metadata was missing.
    InvalidMetadata { field: &'static str },
    /// The persisted or supplied feature schema is incompatible.
    FeatureVersionMismatch { actual: String },
    /// Feature extraction/schema validation failed.
    Feature(FeatureError),
    /// Label validation failed.
    Labels(DcgDatasetError),
    /// Generic dataset conversion failed.
    Dataset(DatasetError),
    /// Duplicate portable record identity was found rather than silently dropped.
    DuplicateRecordId { record_id: String },
    /// At least three independent split groups are needed for three partitions.
    InsufficientSplitGroups { actual: usize },
    /// A requested challenge protocol has no reserved matching records.
    NoChallengeRecords { protocol: String },
    /// File operations failed.
    Io { message: String },
    /// The JSON artifact was malformed or missing required fields.
    Serialization { message: String },
}

impl fmt::Display for PreparedDatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMetadata { field } => {
                write!(f, "Prepared dataset metadata is missing: {field}")
            }
            Self::FeatureVersionMismatch { actual } => write!(
                f,
                "Prepared dataset feature version mismatch: expected {DCG_FEATURE_VERSION}, got {actual}"
            ),
            Self::Feature(error) => write!(f, "Prepared dataset feature error: {error}"),
            Self::Labels(error) => write!(f, "Prepared dataset label error: {error}"),
            Self::Dataset(error) => write!(f, "Prepared dataset conversion error: {error}"),
            Self::DuplicateRecordId { record_id } => {
                write!(f, "Duplicate prepared record ID: {record_id}")
            }
            Self::InsufficientSplitGroups { actual } => write!(
                f,
                "Family-aware splitting requires at least three groups, got {actual}"
            ),
            Self::NoChallengeRecords { protocol } => {
                write!(f, "No reserved challenge records for protocol: {protocol}")
            }
            Self::Io { message } => write!(f, "Prepared dataset IO error: {message}"),
            Self::Serialization { message } => {
                write!(f, "Invalid prepared dataset JSON: {message}")
            }
        }
    }
}

impl Error for PreparedDatasetError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Feature(error) => Some(error),
            Self::Labels(error) => Some(error),
            Self::Dataset(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FeatureError> for PreparedDatasetError {
    fn from(error: FeatureError) -> Self {
        Self::Feature(error)
    }
}
impl From<DcgDatasetError> for PreparedDatasetError {
    fn from(error: DcgDatasetError) -> Self {
        Self::Labels(error)
    }
}
impl From<DatasetError> for PreparedDatasetError {
    fn from(error: DatasetError) -> Self {
        Self::Dataset(error)
    }
}

impl PreparedDcgDataset {
    /// Creates a portable dataset after validating all records.
    pub fn new(
        dataset_version: impl Into<String>,
        records: Vec<PreparedDcgRecord>,
    ) -> Result<Self, PreparedDatasetError> {
        let dataset_version = dataset_version.into();
        if dataset_version.trim().is_empty() {
            return Err(PreparedDatasetError::InvalidMetadata {
                field: "dataset_version",
            });
        }
        if records.is_empty() {
            return Err(PreparedDatasetError::Dataset(DatasetError::EmptyDataset));
        }
        let mut record_ids = BTreeSet::new();
        for record in &records {
            validate_record(record)?;
            if !record_ids.insert(record.record_id.clone()) {
                return Err(PreparedDatasetError::DuplicateRecordId {
                    record_id: record.record_id.clone(),
                });
            }
        }
        let feature_version = &records[0].feature_version;
        if records
            .iter()
            .any(|record| record.feature_version != *feature_version)
        {
            return Err(PreparedDatasetError::FeatureVersionMismatch {
                actual: "mixed feature versions".to_owned(),
            });
        }
        Ok(Self {
            dataset_version,
            benchmark_ready: None,
            records,
        })
    }

    /// Builds self-contained, deterministic fixture records for development and testing.
    pub fn from_fixtures(
        dataset_version: impl Into<String>,
        fixtures: &[DcgFixture],
    ) -> Result<Self, PreparedDatasetError> {
        let records = fixtures
            .iter()
            .map(PreparedDcgRecord::from_fixture)
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(dataset_version, records)
    }

    /// Returns the portable dataset version.
    pub fn dataset_version(&self) -> &str {
        &self.dataset_version
    }

    /// Returns the persisted corpus-design decision, if this artifact has
    /// been explicitly stamped after its readiness audit.
    pub const fn benchmark_ready(&self) -> Option<bool> {
        self.benchmark_ready
    }

    /// Refuses evaluation for legacy, unstamped, or explicitly unready data.
    /// This is intentionally separate from a recomputed audit: the runner
    /// must consume an explicit decision carried by the exact corpus bytes it
    /// is about to train on.
    pub fn require_benchmark_ready(&self) -> Result<(), String> {
        match self.benchmark_ready {
            Some(true) => Ok(()),
            Some(false) => Err(
                "benchmark_ready=false; the corpus failed its persisted readiness gate and cannot be evaluated"
                    .to_owned(),
            ),
            None => Err(
                "benchmark_ready is missing; evaluate only a corpus stamped after its readiness audit"
                    .to_owned(),
            ),
        }
    }

    /// Stamps the explicit corpus decision from the same generalization audit
    /// used by generation. This does not alter labels, features, or records.
    pub fn stamp_benchmark_readiness(
        &mut self,
        promotions: Option<&OracleInvariantPromotionManifest>,
    ) -> GeneralizationReadinessReport {
        let report = promotions.map_or_else(
            || self.generalization_readiness(),
            |manifest| self.generalization_readiness_with_promotions(manifest),
        );
        self.benchmark_ready = Some(report.ready_for_generalization_benchmark);
        report
    }

    /// Returns all validated records.
    pub fn records(&self) -> &[PreparedDcgRecord] {
        &self.records
    }

    /// Returns the one validated feature schema used by every record.
    pub fn feature_version(&self) -> &str {
        &self.records[0].feature_version
    }

    /// Returns the number of prepared records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Reports class, provenance, and family diversity without training a model.
    pub fn statistics(&self) -> DatasetStatistics {
        let mut safe = 0;
        let mut warning = 0;
        let mut breaking = 0;
        let mut families = BTreeSet::new();
        let mut policies = BTreeSet::new();
        let mut mutations = BTreeSet::new();
        for record in &self.records {
            families.insert(&record.split_group_id);
            policies.insert(&record.policy_pack);
            if let Some(generation) = &record.generation {
                mutations.insert(&generation.declared_mutation);
            }
            match record.compatibility_label {
                Some(CompatibilityLabel::Safe) => safe += 1,
                Some(CompatibilityLabel::Warning) => warning += 1,
                Some(CompatibilityLabel::Breaking) => breaking += 1,
                None if record.labels.breaking_change == 1.0 => breaking += 1,
                None => safe += 1,
            }
        }
        DatasetStatistics {
            total_records: self.len(),
            independent_families: families.len(),
            safe,
            warning,
            breaking,
            policy_packs: policies.len(),
            mutation_categories: mutations.len(),
            feature_version: self.feature_version().to_owned(),
        }
    }

    /// Returns the reserved, family-isolated data for a named challenge. Normal
    /// splitting never includes these records. Selection fails explicitly when
    /// a requested challenge was not generated rather than silently falling
    /// back to train/validation/test data.
    pub fn challenge_subset(
        &self,
        protocol: &ChallengeProtocol,
    ) -> Result<Self, PreparedDatasetError> {
        let mut records = self
            .records
            .iter()
            .filter(|record| record.dataset_role == DatasetRole::Challenge)
            .filter(|record| match protocol {
                ChallengeProtocol::HeldOutPolicy { policy_pack } => {
                    record.policy_pack == *policy_pack
                }
                ChallengeProtocol::HeldOutMutation { mutation } => record
                    .generation
                    .as_ref()
                    .is_some_and(|generation| generation.declared_mutation == *mutation),
                ChallengeProtocol::SameMutationDifferentPolicy => false,
            })
            .cloned()
            .collect::<Vec<_>>();
        if matches!(protocol, ChallengeProtocol::SameMutationDifferentPolicy) {
            let mut policies_by_pair = BTreeMap::<(String, String), BTreeSet<String>>::new();
            for record in self
                .records
                .iter()
                .filter(|record| record.dataset_role == DatasetRole::Challenge)
            {
                if let Some(generation) = &record.generation {
                    policies_by_pair
                        .entry((
                            record.family_id.clone(),
                            generation.declared_mutation.clone(),
                        ))
                        .or_default()
                        .insert(record.policy_pack.clone());
                }
            }
            records = self
                .records
                .iter()
                .filter(|record| record.dataset_role == DatasetRole::Challenge)
                .filter(|record| {
                    record.generation.as_ref().is_some_and(|generation| {
                        policies_by_pair
                            .get(&(
                                record.family_id.clone(),
                                generation.declared_mutation.clone(),
                            ))
                            .is_some_and(|policies| policies.len() >= 2)
                    })
                })
                .cloned()
                .collect();
        }
        if records.is_empty() {
            return Err(PreparedDatasetError::NoChallengeRecords {
                protocol: protocol.as_str(),
            });
        }
        Self::new(self.dataset_version.clone(), records)
    }

    /// Audits policy/mutation label shortcuts, family isolation, and the
    /// counterfactual coverage required before challenge metrics are trusted.
    pub fn generalization_readiness(&self) -> GeneralizationReadinessReport {
        self.generalization_readiness_with_promotions(&OracleInvariantPromotionManifest::empty())
    }

    /// As above, but accepts only explicit promotions whose JAR hash,
    /// policy-pack hash, compatibility mode, policy, mutation, and observed
    /// label match each generated record. Any identity change makes evidence
    /// stale automatically and therefore unusable.
    pub fn generalization_readiness_with_promotions(
        &self,
        promotions: &OracleInvariantPromotionManifest,
    ) -> GeneralizationReadinessReport {
        let mut policy_by_label = BTreeMap::new();
        let mut mutation_by_label = BTreeMap::new();
        let mut policy_mutation_by_label = BTreeMap::new();
        let mut labels_by_mutation = BTreeMap::<String, BTreeSet<String>>::new();
        let mut mutations_by_policy = BTreeMap::<String, BTreeSet<String>>::new();
        let mut policies_by_mutation = BTreeMap::<String, BTreeSet<String>>::new();
        let mut rationales_by_mutation = BTreeMap::<String, BTreeSet<String>>::new();
        let mut generated_records_by_mutation = BTreeMap::<String, usize>::new();
        let mut promoted_records_by_mutation = BTreeMap::<String, usize>::new();
        let mut policies_by_pair = BTreeMap::<(String, String), BTreeSet<String>>::new();
        let mut standard_families = BTreeSet::new();
        let mut challenge_families = BTreeSet::new();
        let mut standard_pairs = BTreeSet::new();
        let mut challenge_pairs = BTreeSet::new();
        let mut optional_profile_standard_families = BTreeMap::<String, BTreeSet<String>>::new();
        let mut optional_profile_challenge_families = BTreeMap::<String, BTreeSet<String>>::new();
        let mut optional_records = 0usize;
        let mut optional_records_without_profile = 0usize;
        let mut challenge_records = 0;
        let mut empty_breaking_decisions = 0usize;

        for record in &self.records {
            let label = record_label(record).to_owned();
            increment_distribution(&mut policy_by_label, &record.policy_pack, &label);
            if record.dataset_role == DatasetRole::Challenge {
                challenge_records += 1;
                challenge_families.insert(record.family_id.clone());
            } else {
                standard_families.insert(record.family_id.clone());
            }
            if let Some(generation) = &record.generation {
                if generation.declared_mutation == "optional_field_added" {
                    optional_records += 1;
                    match generation.root_object_profile.as_deref() {
                        Some("open") => match record.dataset_role {
                            DatasetRole::Standard => {
                                optional_profile_standard_families
                                    .entry("open".to_owned())
                                    .or_default()
                                    .insert(record.family_id.clone());
                            }
                            DatasetRole::Challenge => {
                                optional_profile_challenge_families
                                    .entry("open".to_owned())
                                    .or_default()
                                    .insert(record.family_id.clone());
                            }
                        },
                        Some("closed") => match record.dataset_role {
                            DatasetRole::Standard => {
                                optional_profile_standard_families
                                    .entry("closed".to_owned())
                                    .or_default()
                                    .insert(record.family_id.clone());
                            }
                            DatasetRole::Challenge => {
                                optional_profile_challenge_families
                                    .entry("closed".to_owned())
                                    .or_default()
                                    .insert(record.family_id.clone());
                            }
                        },
                        _ => optional_records_without_profile += 1,
                    }
                }
                // A real pinned-JAR compatibility failure emits its FAIL
                // decision on stdout. An empty stream cannot substantiate a
                // retained BREAKING label; it may be an older artifact made
                // before uncaught exit-1 JVM failures were rejected.
                if generation.oracle_outcome == "breaking"
                    && generation.oracle_stdout.trim().is_empty()
                {
                    empty_breaking_decisions += 1;
                }
                increment_distribution(
                    &mut mutation_by_label,
                    &generation.declared_mutation,
                    &label,
                );
                increment_distribution(
                    &mut policy_mutation_by_label,
                    &format!("{}:{}", record.policy_pack, generation.declared_mutation),
                    &label,
                );
                labels_by_mutation
                    .entry(generation.declared_mutation.clone())
                    .or_default()
                    .insert(label.clone());
                mutations_by_policy
                    .entry(record.policy_pack.clone())
                    .or_default()
                    .insert(generation.declared_mutation.clone());
                policies_by_mutation
                    .entry(generation.declared_mutation.clone())
                    .or_default()
                    .insert(record.policy_pack.clone());
                *generated_records_by_mutation
                    .entry(generation.declared_mutation.clone())
                    .or_default() += 1;
                if let Some(mode) = OracleCompatibilityMode::parse(&generation.compatibility_mode)
                    && let Some(promotion) = promotions.matching(
                        &generation.declared_mutation,
                        mode,
                        &record.policy_pack,
                        &label,
                        &generation.oracle_jar_sha256,
                        &generation.policy_packs_sha256,
                    )
                {
                    *promoted_records_by_mutation
                        .entry(generation.declared_mutation.clone())
                        .or_default() += 1;
                    rationales_by_mutation
                        .entry(generation.declared_mutation.clone())
                        .or_default()
                        .insert(promotion.rationale.clone());
                }
                policies_by_pair
                    .entry((
                        record.family_id.clone(),
                        generation.declared_mutation.clone(),
                    ))
                    .or_default()
                    .insert(record.policy_pack.clone());
                if !generation.pair_fingerprint.is_empty() {
                    match record.dataset_role {
                        DatasetRole::Standard => {
                            standard_pairs.insert(generation.pair_fingerprint.clone());
                        }
                        DatasetRole::Challenge => {
                            challenge_pairs.insert(generation.pair_fingerprint.clone());
                        }
                    }
                }
            }
        }

        let oracle_invariant_exceptions = labels_by_mutation
            .iter()
            .filter_map(|(mutation, labels)| {
                let rationale = rationales_by_mutation.get(mutation)?;
                (labels.len() == 1
                    && rationale.len() == 1
                    && promoted_records_by_mutation
                        .get(mutation)
                        .copied()
                        .unwrap_or(0)
                        == generated_records_by_mutation
                            .get(mutation)
                            .copied()
                            .unwrap_or(0))
                .then(|| OracleInvariantException {
                    mutation: mutation.clone(),
                    observed_policies: policies_by_mutation
                        .get(mutation)
                        .map(|policies| policies.iter().cloned().collect())
                        .unwrap_or_default(),
                    observed_labels: mutation_by_label.get(mutation).cloned().unwrap_or_default(),
                    rationale: rationale.iter().next().cloned().unwrap_or_default(),
                })
            })
            .collect::<Vec<_>>();
        let observed_oracle_invariant_mutations = oracle_invariant_exceptions
            .iter()
            .map(|exception| exception.mutation.clone())
            .collect::<Vec<_>>();
        let invariant = observed_oracle_invariant_mutations
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut shortcut_risks = Vec::new();
        for (policy, labels) in &policy_by_label {
            let has_non_invariant_mutation =
                mutations_by_policy.get(policy).is_some_and(|mutations| {
                    mutations
                        .iter()
                        .any(|mutation| !invariant.contains(mutation))
                });
            if labels.len() == 1 && has_non_invariant_mutation {
                shortcut_risks.push(format!("policy `{policy}` is label-pure"));
            }
        }
        for (mutation, labels) in &labels_by_mutation {
            if labels.len() == 1 && !invariant.contains(mutation) {
                shortcut_risks.push(format!(
                    "mutation `{mutation}` is label-pure without a valid promoted oracle-invariant exception"
                ));
            }
            if policies_by_mutation
                .get(mutation)
                .is_some_and(|policies| policies.len() < 3)
            {
                shortcut_risks.push(format!(
                    "mutation `{mutation}` is represented under fewer than three policies"
                ));
            }
        }
        let complete_policy_counterfactual_pairs = policies_by_pair
            .values()
            .filter(|policies| policies.len() >= 3)
            .count();
        let family_mutation_pairs = policies_by_pair.len();
        let challenge_family_leakage = !standard_families.is_disjoint(&challenge_families);
        let challenge_pair_leakage = !standard_pairs.is_disjoint(&challenge_pairs);
        let challenge_protocols = self.challenge_protocol_reports();
        let optional_field_profile_coverage = ["open", "closed"]
            .into_iter()
            .map(|profile| {
                (
                    profile.to_owned(),
                    OptionalFieldProfileCoverage {
                        standard_families: optional_profile_standard_families
                            .get(profile)
                            .map_or(0, BTreeSet::len),
                        challenge_families: optional_profile_challenge_families
                            .get(profile)
                            .map_or(0, BTreeSet::len),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut reasons = Vec::new();
        if empty_breaking_decisions > 0 {
            reasons.push(format!(
                "{empty_breaking_decisions} retained BREAKING records have no oracle decision stdout; regenerate them with the fatal-oracle-failure guard"
            ));
        }
        if standard_families.len() < 3 {
            reasons.push(
                "fewer than three standard families remain for train/validation/test".to_owned(),
            );
        }
        if challenge_records == 0 {
            reasons.push("no complete families are reserved for challenge evaluation".to_owned());
        }
        if challenge_family_leakage {
            reasons.push("a family occurs in both standard and challenge roles".to_owned());
        }
        if challenge_pair_leakage {
            reasons.push(
                "an oracle pair fingerprint occurs in both standard and challenge roles".to_owned(),
            );
        }
        if family_mutation_pairs == 0 {
            reasons.push("no generator mutation provenance is available".to_owned());
        } else if complete_policy_counterfactual_pairs == 0 {
            reasons.push(
                "no family/mutation pair is observed under all three policy packs".to_owned(),
            );
        }
        if optional_records > 0 {
            if optional_records_without_profile > 0 {
                reasons.push(format!(
                    "{optional_records_without_profile} optional-field records lack open/closed root-profile provenance"
                ));
            }
            for (profile, coverage) in &optional_field_profile_coverage {
                if coverage.standard_families == 0 {
                    reasons.push(format!(
                        "optional-field structural coverage has no `{profile}` standard families"
                    ));
                }
                if coverage.challenge_families == 0 {
                    reasons.push(format!(
                        "optional-field structural coverage has no `{profile}` challenge families"
                    ));
                }
            }
        }
        if !shortcut_risks.is_empty() {
            reasons.push(
                "label shortcuts remain outside observed oracle-invariant mutation exceptions"
                    .to_owned(),
            );
        }
        let required_challenge_protocol_is_unmeaningful =
            challenge_protocols.iter().any(|report| {
                if report.meaningful {
                    return false;
                }
                match report.protocol.strip_prefix("held-out-mutation:") {
                    // A mutation observed as oracle-invariant is intentionally
                    // retained but cannot be a multi-class mutation challenge.
                    Some(mutation) => !invariant.contains(mutation),
                    None => true,
                }
            });
        if required_challenge_protocol_is_unmeaningful {
            reasons.push("a required challenge protocol lacks multi-class coverage".to_owned());
        }
        GeneralizationReadinessReport {
            policy_by_label,
            mutation_by_label,
            policy_mutation_by_label,
            observed_oracle_invariant_mutations,
            oracle_invariant_exceptions,
            shortcut_risks,
            challenge_records,
            standard_families: standard_families.len(),
            challenge_families: challenge_families.len(),
            challenge_family_leakage,
            challenge_pair_leakage,
            family_mutation_pairs,
            complete_policy_counterfactual_pairs,
            optional_field_profile_coverage,
            challenge_protocols,
            ready_for_generalization_benchmark: reasons.is_empty(),
            reasons,
        }
    }

    fn challenge_protocol_reports(&self) -> Vec<ChallengeProtocolReport> {
        let mut protocols = Vec::new();
        let policies = self
            .records
            .iter()
            .filter(|record| record.dataset_role == DatasetRole::Challenge)
            .map(|record| record.policy_pack.clone())
            .collect::<BTreeSet<_>>();
        let mutations = self
            .records
            .iter()
            .filter(|record| record.dataset_role == DatasetRole::Challenge)
            .filter_map(|record| {
                record
                    .generation
                    .as_ref()
                    .map(|g| g.declared_mutation.clone())
            })
            .collect::<BTreeSet<_>>();
        for policy_pack in policies {
            protocols.push(
                self.challenge_protocol_report(&ChallengeProtocol::HeldOutPolicy { policy_pack }),
            );
        }
        for mutation in mutations {
            protocols.push(
                self.challenge_protocol_report(&ChallengeProtocol::HeldOutMutation { mutation }),
            );
        }
        protocols
            .push(self.challenge_protocol_report(&ChallengeProtocol::SameMutationDifferentPolicy));
        protocols
    }

    fn challenge_protocol_report(&self, protocol: &ChallengeProtocol) -> ChallengeProtocolReport {
        match self.challenge_subset(protocol) {
            Ok(dataset) => {
                let labels = label_distribution(dataset.records());
                let families = dataset
                    .records
                    .iter()
                    .map(|r| &r.family_id)
                    .collect::<BTreeSet<_>>()
                    .len();
                let policies = dataset
                    .records
                    .iter()
                    .map(|r| &r.policy_pack)
                    .collect::<BTreeSet<_>>()
                    .len();
                let mutations = dataset
                    .records
                    .iter()
                    .filter_map(|r| r.generation.as_ref().map(|g| &g.declared_mutation))
                    .collect::<BTreeSet<_>>()
                    .len();
                let meaningful = labels.len() >= 2;
                ChallengeProtocolReport {
                    protocol: protocol.as_str(),
                    records: dataset.len(),
                    families,
                    policies,
                    mutations,
                    labels,
                    meaningful,
                    reason: (!meaningful)
                        .then(|| "only one oracle label is represented".to_owned()),
                }
            }
            Err(error) => ChallengeProtocolReport {
                protocol: protocol.as_str(),
                records: 0,
                families: 0,
                policies: 0,
                mutations: 0,
                labels: BTreeMap::new(),
                meaningful: false,
                reason: Some(error.to_string()),
            },
        }
    }

    /// Checks the minimum engineering conditions for a grouped training experiment.
    pub fn training_readiness(
        &self,
        target_mode: TargetMode,
        split: DatasetSplitConfig,
    ) -> TrainingReadinessReport {
        let statistics = self.statistics();
        let standard_records = self
            .records
            .iter()
            .filter(|record| record.dataset_role == DatasetRole::Standard)
            .cloned()
            .collect::<Vec<_>>();
        let standard_families = standard_records
            .iter()
            .map(|record| &record.split_group_id)
            .collect::<BTreeSet<_>>()
            .len();
        let standard_classes = class_counts(&standard_records);
        let mut reasons = Vec::new();
        if standard_families < 3 {
            reasons
                .push("fewer than three standard (non-challenge) independent families".to_owned());
        }
        if standard_classes[CompatibilityLabel::Breaking.class_index() as usize] == 0
            || standard_classes[CompatibilityLabel::Safe.class_index() as usize] == 0
        {
            reasons.push(
                "binary BREAKING/non-breaking class is missing from standard records".to_owned(),
            );
        }
        if matches!(target_mode, TargetMode::ThreeWayCompatibility) && standard_classes.contains(&0)
        {
            reasons.push(
                "one or more SAFE/WARNING/BREAKING classes are missing from standard records"
                    .to_owned(),
            );
        }
        let split_result = self.split_by_group(split);
        let (partition_records, partition_families, leakage_detected) = match split_result {
            Ok(partitions) => {
                let records = (
                    partitions.train.len(),
                    partitions.validation.len(),
                    partitions.test.len(),
                );
                let family_count = |dataset: &PreparedDcgDataset| {
                    dataset
                        .records
                        .iter()
                        .map(|record| &record.split_group_id)
                        .collect::<BTreeSet<_>>()
                        .len()
                };
                let groups = (
                    family_count(&partitions.train),
                    family_count(&partitions.validation),
                    family_count(&partitions.test),
                );
                let ids = [
                    partitions.train(),
                    partitions.validation(),
                    partitions.test(),
                ]
                .map(|dataset| {
                    dataset
                        .records
                        .iter()
                        .map(|record| record.split_group_id.as_str())
                        .collect::<BTreeSet<_>>()
                });
                let leaks = ids[0].intersection(&ids[1]).next().is_some()
                    || ids[0].intersection(&ids[2]).next().is_some()
                    || ids[1].intersection(&ids[2]).next().is_some();
                (Some(records), Some(groups), leaks)
            }
            Err(error) => {
                reasons.push(error.to_string());
                (None, None, true)
            }
        };
        if leakage_detected {
            reasons.push("family leakage detected or split could not be constructed".to_owned());
        }
        TrainingReadinessReport {
            statistics,
            target_mode,
            partition_records,
            partition_families,
            leakage_detected,
            ready_for_training: reasons.is_empty(),
            reasons,
        }
    }

    /// Returns true when this dataset has no records.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Converts records into the generic supervised dataset for one model task.
    pub fn to_dataset(&self, kind: PredictionKind) -> Result<Dataset, PreparedDatasetError> {
        Dataset::new(
            self.records
                .iter()
                .map(|record| record.features.clone())
                .collect(),
            self.records
                .iter()
                .map(|record| record.labels.target_for(kind))
                .collect(),
        )
        .map_err(Into::into)
    }

    /// Converts canonical oracle labels into either the binary or one-hot
    /// three-way target view. Three-way training deliberately requires the
    /// original oracle category rather than inferring WARNING from risk.
    pub fn to_target_dataset(
        &self,
        target_mode: TargetMode,
    ) -> Result<Dataset, PreparedDatasetError> {
        let targets = self
            .records
            .iter()
            .map(|record| match (target_mode, record.compatibility_label) {
                (TargetMode::BinaryBreaking, Some(label)) => Ok(target_mode.target_for(label)),
                (TargetMode::BinaryBreaking, None) => {
                    Ok(Vector::new(vec![record.labels.breaking_change]))
                }
                (TargetMode::ThreeWayCompatibility, Some(label)) => {
                    Ok(target_mode.target_for(label))
                }
                (TargetMode::ThreeWayCompatibility, None) => {
                    Err(PreparedDatasetError::InvalidMetadata {
                        field: "compatibility_label",
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        Dataset::new(
            self.records
                .iter()
                .map(|record| record.features.clone())
                .collect(),
            targets,
        )
        .map_err(Into::into)
    }

    /// Splits complete policy/contract families together using a seeded stable ordering.
    pub fn split_by_group(
        &self,
        config: DatasetSplitConfig,
    ) -> Result<PreparedDcgDatasetSplit, PreparedDatasetError> {
        DatasetSplitConfig::new(
            config.train_ratio,
            config.validation_ratio,
            config.test_ratio,
            config.seed,
        )?;
        let mut groups = BTreeMap::<String, Vec<PreparedDcgRecord>>::new();
        for record in &self.records {
            if record.dataset_role == DatasetRole::Challenge {
                continue;
            }
            groups
                .entry(record.split_group_id.clone())
                .or_default()
                .push(record.clone());
        }
        if groups.len() < 3 {
            return Err(PreparedDatasetError::InsufficientSplitGroups {
                actual: groups.len(),
            });
        }
        let mut groups = groups.into_iter().collect::<Vec<_>>();
        // Allocate the class-rich, large families first. Otherwise hundreds of
        // one-record WARNING-only families can consume the training target
        // before the BREAKING-bearing families are considered.
        groups.sort_by(|(left_id, left_records), (right_id, right_records)| {
            right_records.len().cmp(&left_records.len()).then_with(|| {
                stable_group_hash(left_id, config.seed)
                    .cmp(&stable_group_hash(right_id, config.seed))
            })
        });

        let ratios = [
            config.train_ratio,
            config.validation_ratio,
            config.test_ratio,
        ];
        let targets = ratios.map(|ratio| ratio * self.len() as f64);
        let total_class_counts = class_counts(&self.records);
        let class_targets =
            ratios.map(|ratio| total_class_counts.map(|count| ratio * count as f64));
        let mut partitions = [Vec::new(), Vec::new(), Vec::new()];
        let mut partition_sizes = [0usize; 3];
        let mut partition_class_counts = [[0usize; 3]; 3];
        let total_groups = groups.len();
        for (group_index, (_, records)) in groups.into_iter().enumerate() {
            let group_counts = class_counts(&records);
            let empty_partitions = (0..3)
                .filter(|partition| partitions[*partition].is_empty())
                .collect::<Vec<_>>();
            let remaining_groups = total_groups - group_index;
            let candidates = if remaining_groups <= empty_partitions.len() {
                empty_partitions
            } else {
                vec![0, 1, 2]
            };
            let partition_index = candidates
                .into_iter()
                .min_by(|left, right| {
                    split_balance_score(
                        *left,
                        records.len(),
                        group_counts,
                        &partition_sizes,
                        &partition_class_counts,
                        targets,
                        class_targets,
                    )
                    .total_cmp(&split_balance_score(
                        *right,
                        records.len(),
                        group_counts,
                        &partition_sizes,
                        &partition_class_counts,
                        targets,
                        class_targets,
                    ))
                })
                .unwrap_or(0);
            partition_sizes[partition_index] += records.len();
            for class in 0..3 {
                partition_class_counts[partition_index][class] += group_counts[class];
            }
            partitions[partition_index].extend(records);
        }
        Ok(PreparedDcgDatasetSplit {
            train: Self::new(self.dataset_version.clone(), partitions[0].clone())?,
            validation: Self::new(self.dataset_version.clone(), partitions[1].clone())?,
            test: Self::new(self.dataset_version.clone(), partitions[2].clone())?,
        })
    }

    /// Writes a portable JSON prepared-dataset artifact without local paths.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), PreparedDatasetError> {
        let file = PreparedDatasetFile::from_dataset(self);
        let contents = serde_json::to_vec_pretty(&file).map_err(|error| {
            PreparedDatasetError::Serialization {
                message: error.to_string(),
            }
        })?;
        write_atomically(path.as_ref(), &contents)
    }

    /// Loads a portable JSON prepared-dataset artifact and validates every record.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, PreparedDatasetError> {
        let contents = fs::read(path).map_err(|error| PreparedDatasetError::Io {
            message: error.to_string(),
        })?;
        serde_json::from_slice::<PreparedDatasetFile>(&contents)
            .map_err(|error| PreparedDatasetError::Serialization {
                message: error.to_string(),
            })?
            .into_dataset()
    }
}

fn class_counts(records: &[PreparedDcgRecord]) -> [usize; 3] {
    let mut counts = [0; 3];
    for record in records {
        let class = match record.compatibility_label {
            Some(label) => label.class_index() as usize,
            None if record.labels.breaking_change == 1.0 => {
                CompatibilityLabel::Breaking.class_index() as usize
            }
            None => CompatibilityLabel::Safe.class_index() as usize,
        };
        counts[class] += 1;
    }
    counts
}

fn record_label(record: &PreparedDcgRecord) -> &'static str {
    match record.compatibility_label {
        Some(label) => label.as_str(),
        None if record.labels.breaking_change == 1.0 => CompatibilityLabel::Breaking.as_str(),
        None => CompatibilityLabel::Safe.as_str(),
    }
}

fn increment_distribution(
    distribution: &mut BTreeMap<String, BTreeMap<String, usize>>,
    key: &str,
    label: &str,
) {
    *distribution
        .entry(key.to_owned())
        .or_default()
        .entry(label.to_owned())
        .or_default() += 1;
}

fn label_distribution(records: &[PreparedDcgRecord]) -> BTreeMap<String, usize> {
    let mut labels = BTreeMap::new();
    for record in records {
        *labels.entry(record_label(record).to_owned()).or_default() += 1;
    }
    labels
}

fn split_balance_score(
    partition: usize,
    incoming_records: usize,
    incoming_classes: [usize; 3],
    partition_sizes: &[usize; 3],
    partition_class_counts: &[[usize; 3]; 3],
    record_targets: [f64; 3],
    class_targets: [[f64; 3]; 3],
) -> f64 {
    let size = partition_sizes[partition] + incoming_records;
    // Compare the *change* in global balance error, not the candidate
    // partition's absolute error in isolation. The latter incorrectly fills
    // small validation/test targets first. A negative score is an improvement.
    let previous_size = partition_sizes[partition] as f64;
    let target_size = record_targets[partition].max(1.0);
    let mut score = (size as f64 - target_size).powi(2) / target_size
        - (previous_size - target_size).powi(2) / target_size;
    for class in 0..3 {
        let previous_count = partition_class_counts[partition][class] as f64;
        let count = previous_count + incoming_classes[class] as f64;
        let target_count = class_targets[partition][class].max(1.0);
        score += (count - target_count).powi(2) / target_count
            - (previous_count - target_count).powi(2) / target_count;
    }
    score
}

impl PreparedDcgDatasetSplit {
    /// Returns the training partition, the only partition used to fit a scaler.
    pub fn train(&self) -> &PreparedDcgDataset {
        &self.train
    }
    /// Returns the validation partition used for model and threshold evaluation.
    pub fn validation(&self) -> &PreparedDcgDataset {
        &self.validation
    }
    /// Returns the final isolated test partition.
    pub fn test(&self) -> &PreparedDcgDataset {
        &self.test
    }
}

impl PreparedDcgRecord {
    /// Converts one self-contained fixture to a portable policy-labelled record.
    pub fn from_fixture(fixture: &DcgFixture) -> Result<Self, PreparedDatasetError> {
        let breaking = f64::from(fixture.deterministic_breaking);
        let incompatible = f64::from(matches!(
            fixture.change.compatibility_status,
            crate::features::CompatibilityStatus::Incompatible
        ));
        Ok(Self {
            record_id: fixture.id.to_owned(),
            source: "self-contained-dcg-fixture".to_owned(),
            family_id: fixture.contract_family.to_owned(),
            split_group_id: fixture.contract_family.to_owned(),
            dataset_role: DatasetRole::Standard,
            contract_id: fixture.contract_family.to_owned(),
            old_version: version_name(fixture.old_version),
            new_version: version_name(fixture.new_version),
            policy_pack: policy_pack_name(fixture.policy_pack).to_owned(),
            feature_version: DCG_FEATURE_VERSION.to_owned(),
            features: ContractFeatureExtractor::new().extract(&fixture.change)?,
            labels: ContractLabels {
                breaking_change: breaking,
                incompatible,
                risk_score: if fixture.deterministic_breaking {
                    0.8
                } else {
                    0.15
                },
            },
            generation: None,
            compatibility_label: None,
        })
    }
}

fn validate_record(record: &PreparedDcgRecord) -> Result<(), PreparedDatasetError> {
    for (field, value) in [
        ("record_id", &record.record_id),
        ("source", &record.source),
        ("family_id", &record.family_id),
        ("split_group_id", &record.split_group_id),
        ("contract_id", &record.contract_id),
        ("old_version", &record.old_version),
        ("new_version", &record.new_version),
        ("policy_pack", &record.policy_pack),
    ] {
        if value.trim().is_empty() {
            return Err(PreparedDatasetError::InvalidMetadata { field });
        }
    }
    if record.feature_version != DCG_FEATURE_VERSION
        && record.feature_version != DCG_FEATURE_V2_VERSION
        && record.feature_version != DCG_FEATURE_V3_VERSION
        && record.feature_version != DCG_FEATURE_V4_VERSION
        && record.feature_version != DCG_FEATURE_V5_VERSION
        && record.feature_version != DCG_FEATURE_V6_VERSION
    {
        return Err(PreparedDatasetError::FeatureVersionMismatch {
            actual: record.feature_version.clone(),
        });
    }
    validate_feature_vector_for_version(&record.feature_version, &record.features)?;
    record.labels.validate()?;
    if let Some(label) = record.compatibility_label
        && (record.labels.breaking_change != label.binary_breaking()
            || record.labels.incompatible != label.binary_breaking())
    {
        return Err(PreparedDatasetError::InvalidMetadata {
            field: "compatibility_label",
        });
    }
    if let Some(generation) = &record.generation {
        for (field, value) in [
            ("oracle_jar_sha256", &generation.oracle_jar_sha256),
            ("oracle_outcome", &generation.oracle_outcome),
            ("declared_mutation", &generation.declared_mutation),
        ] {
            if value.trim().is_empty() {
                return Err(PreparedDatasetError::InvalidMetadata { field });
            }
        }
        if !matches!(
            generation.oracle_outcome.as_str(),
            "safe" | "warning" | "breaking" | "pass"
        ) {
            return Err(PreparedDatasetError::InvalidMetadata {
                field: "oracle_outcome",
            });
        }
        if generation
            .root_object_profile
            .as_deref()
            .is_some_and(|profile| !matches!(profile, "open" | "closed"))
        {
            return Err(PreparedDatasetError::InvalidMetadata {
                field: "root_object_profile",
            });
        }
    }
    Ok(())
}

fn stable_group_hash(value: &str, seed: u64) -> u64 {
    value.bytes().fold(seed ^ 0xcbf29ce484222325, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(0x100000001b3)
    })
}

fn policy_pack_name(pack: FixturePolicyPack) -> &'static str {
    match pack {
        FixturePolicyPack::Baseline => "baseline",
        FixturePolicyPack::Strict => "strict",
        FixturePolicyPack::Relaxed => "relaxed",
    }
}

fn version_name(version: crate::features::SemanticVersion) -> String {
    format!("{}.{}.{}", version.major, version.minor, version.patch)
}

fn write_atomically(path: &Path, contents: &[u8]) -> Result<(), PreparedDatasetError> {
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, contents).map_err(|error| PreparedDatasetError::Io {
        message: error.to_string(),
    })?;
    fs::rename(&temporary, path).map_err(|error| PreparedDatasetError::Io {
        message: error.to_string(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
struct PreparedDatasetFile {
    format_version: String,
    dataset_version: String,
    feature_version: String,
    /// `null` preserves the distinguishable legacy/unstamped state. The
    /// evaluation runner rejects both `null` and `false`.
    #[serde(default)]
    benchmark_ready: Option<bool>,
    records: Vec<PreparedRecordFile>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PreparedRecordFile {
    record_id: String,
    source: String,
    family_id: String,
    split_group_id: String,
    #[serde(default = "default_dataset_role")]
    dataset_role: String,
    contract_id: String,
    old_version: String,
    new_version: String,
    policy_pack: String,
    features: Vec<f64>,
    breaking_change: f64,
    incompatible: f64,
    risk_score: f64,
    #[serde(default)]
    compatibility_label: Option<String>,
    generation: Option<GeneratedRecordProvenanceFile>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeneratedRecordProvenanceFile {
    oracle_jar_sha256: String,
    #[serde(default)]
    policy_packs_sha256: String,
    oracle_outcome: String,
    declared_mutation: String,
    #[serde(default)]
    mutation_variant: String,
    #[serde(default)]
    root_object_profile: Option<String>,
    oracle_stdout: String,
    #[serde(default)]
    compatibility_mode: String,
    #[serde(default)]
    generator_version: String,
    #[serde(default)]
    generation_seed: u64,
    #[serde(default)]
    label_source: String,
    #[serde(default)]
    oracle_invariant_rationale: Option<String>,
    #[serde(default)]
    pair_fingerprint: String,
}

impl PreparedDatasetFile {
    fn from_dataset(dataset: &PreparedDcgDataset) -> Self {
        Self {
            format_version: PREPARED_DATASET_FORMAT_VERSION.to_owned(),
            dataset_version: dataset.dataset_version.clone(),
            feature_version: dataset.feature_version().to_owned(),
            benchmark_ready: dataset.benchmark_ready,
            records: dataset
                .records
                .iter()
                .map(|record| PreparedRecordFile {
                    record_id: record.record_id.clone(),
                    source: record.source.clone(),
                    family_id: record.family_id.clone(),
                    split_group_id: record.split_group_id.clone(),
                    dataset_role: record.dataset_role.as_str().to_owned(),
                    contract_id: record.contract_id.clone(),
                    old_version: record.old_version.clone(),
                    new_version: record.new_version.clone(),
                    policy_pack: record.policy_pack.clone(),
                    features: record.features.iter().copied().collect(),
                    breaking_change: record.labels.breaking_change,
                    incompatible: record.labels.incompatible,
                    risk_score: record.labels.risk_score,
                    compatibility_label: record
                        .compatibility_label
                        .map(|label| label.as_str().to_owned()),
                    generation: record.generation.as_ref().map(|generation| {
                        GeneratedRecordProvenanceFile {
                            oracle_jar_sha256: generation.oracle_jar_sha256.clone(),
                            policy_packs_sha256: generation.policy_packs_sha256.clone(),
                            oracle_outcome: generation.oracle_outcome.clone(),
                            declared_mutation: generation.declared_mutation.clone(),
                            mutation_variant: generation.mutation_variant.clone(),
                            root_object_profile: generation.root_object_profile.clone(),
                            oracle_stdout: generation.oracle_stdout.clone(),
                            compatibility_mode: generation.compatibility_mode.clone(),
                            generator_version: generation.generator_version.clone(),
                            generation_seed: generation.generation_seed,
                            label_source: generation.label_source.clone(),
                            oracle_invariant_rationale: generation
                                .oracle_invariant_rationale
                                .clone(),
                            pair_fingerprint: generation.pair_fingerprint.clone(),
                        }
                    }),
                })
                .collect(),
        }
    }

    fn into_dataset(self) -> Result<PreparedDcgDataset, PreparedDatasetError> {
        if self.format_version != PREPARED_DATASET_FORMAT_VERSION {
            return Err(PreparedDatasetError::Serialization {
                message: format!("unsupported format version: {}", self.format_version),
            });
        }
        if self.feature_version != DCG_FEATURE_VERSION
            && self.feature_version != DCG_FEATURE_V2_VERSION
            && self.feature_version != DCG_FEATURE_V3_VERSION
            && self.feature_version != DCG_FEATURE_V4_VERSION
            && self.feature_version != DCG_FEATURE_V5_VERSION
            && self.feature_version != DCG_FEATURE_V6_VERSION
        {
            return Err(PreparedDatasetError::FeatureVersionMismatch {
                actual: self.feature_version,
            });
        }
        let feature_version = self.feature_version.clone();
        let records = self
            .records
            .into_iter()
            .map(|record| {
                let compatibility_label = match record.compatibility_label {
                    Some(label) => Some(CompatibilityLabel::parse(&label).ok_or(
                        PreparedDatasetError::InvalidMetadata {
                            field: "compatibility_label",
                        },
                    )?),
                    None => None,
                };
                Ok(PreparedDcgRecord {
                    record_id: record.record_id,
                    source: record.source,
                    family_id: record.family_id,
                    split_group_id: record.split_group_id,
                    dataset_role: DatasetRole::parse(&record.dataset_role).ok_or(
                        PreparedDatasetError::InvalidMetadata {
                            field: "dataset_role",
                        },
                    )?,
                    contract_id: record.contract_id,
                    old_version: record.old_version,
                    new_version: record.new_version,
                    policy_pack: record.policy_pack,
                    feature_version: feature_version.clone(),
                    features: Vector::new(record.features),
                    labels: ContractLabels {
                        breaking_change: record.breaking_change,
                        incompatible: record.incompatible,
                        risk_score: record.risk_score,
                    },
                    compatibility_label,
                    generation: record
                        .generation
                        .map(|generation| GeneratedRecordProvenance {
                            oracle_jar_sha256: generation.oracle_jar_sha256,
                            policy_packs_sha256: generation.policy_packs_sha256,
                            oracle_outcome: generation.oracle_outcome,
                            declared_mutation: generation.declared_mutation,
                            mutation_variant: generation.mutation_variant,
                            root_object_profile: generation.root_object_profile,
                            oracle_stdout: generation.oracle_stdout,
                            compatibility_mode: generation.compatibility_mode,
                            generator_version: generation.generator_version,
                            generation_seed: generation.generation_seed,
                            label_source: generation.label_source,
                            oracle_invariant_rationale: generation.oracle_invariant_rationale,
                            pair_fingerprint: generation.pair_fingerprint,
                        }),
                })
            })
            .collect::<Result<Vec<_>, PreparedDatasetError>>()?;
        let mut dataset = PreparedDcgDataset::new(self.dataset_version, records)?;
        dataset.benchmark_ready = self.benchmark_ready;
        Ok(dataset)
    }
}

fn default_dataset_role() -> String {
    DatasetRole::Standard.as_str().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{SchemaChangeFeatureExtractor, realistic_fixtures};

    #[test]
    fn persists_records_and_keeps_families_in_one_partition() {
        let dataset =
            PreparedDcgDataset::from_fixtures("fixture-dataset-v1", &realistic_fixtures()).unwrap();
        let split = dataset
            .split_by_group(DatasetSplitConfig::new(0.7, 0.15, 0.15, 11).unwrap())
            .unwrap();
        for family in ["orders.created", "payments.completed", "analytics.events"] {
            let occurrences = [split.train(), split.validation(), split.test()]
                .iter()
                .filter(|partition| {
                    partition
                        .records()
                        .iter()
                        .any(|record| record.family_id == family)
                })
                .count();
            assert_eq!(occurrences, 1, "{family} leaked across partitions");
        }
        let path = std::env::temp_dir().join(format!("prepared-dcg-{}.json", std::process::id()));
        dataset.save(&path).unwrap();
        assert_eq!(PreparedDcgDataset::load(&path).unwrap(), dataset);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_incompatible_record_versions() {
        let v1_record = PreparedDcgRecord::from_fixture(&realistic_fixtures()[0]).unwrap();
        let mut record = PreparedDcgRecord::from_fixture(&realistic_fixtures()[1]).unwrap();
        record.feature_version = "dcg-features-v2".to_owned();
        record.features = SchemaChangeFeatureExtractor::new()
            .extract(
                r#"{"type":"object","properties":{"name":{"type":"string"}}}"#,
                r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}"#,
                "baseline",
            )
            .unwrap();
        assert!(matches!(
            PreparedDcgDataset::new("v1", vec![v1_record, record]),
            Err(PreparedDatasetError::FeatureVersionMismatch { .. })
        ));
    }

    #[test]
    fn balance_score_fills_the_largest_partition_deficit_first() {
        let score_for_train = split_balance_score(
            0,
            27,
            [4, 1, 22],
            &[0, 0, 0],
            &[[0; 3]; 3],
            [701.4, 150.3, 150.3],
            [
                [233.8, 233.8, 233.8],
                [50.1, 50.1, 50.1],
                [50.1, 50.1, 50.1],
            ],
        );
        let score_for_validation = split_balance_score(
            1,
            27,
            [4, 1, 22],
            &[0, 0, 0],
            &[[0; 3]; 3],
            [701.4, 150.3, 150.3],
            [
                [233.8, 233.8, 233.8],
                [50.1, 50.1, 50.1],
                [50.1, 50.1, 50.1],
            ],
        );
        assert!(score_for_train < score_for_validation);
    }

    fn generated_record(
        id: &str,
        family: &str,
        role: DatasetRole,
        policy: &str,
        mutation: &str,
        label: CompatibilityLabel,
    ) -> PreparedDcgRecord {
        let mut record = PreparedDcgRecord::from_fixture(&realistic_fixtures()[0]).unwrap();
        record.record_id = id.to_owned();
        record.family_id = family.to_owned();
        record.split_group_id = family.to_owned();
        record.contract_id = family.to_owned();
        record.dataset_role = role;
        record.policy_pack = policy.to_owned();
        record.compatibility_label = Some(label);
        record.labels = ContractLabels {
            breaking_change: label.binary_breaking(),
            incompatible: label.binary_breaking(),
            risk_score: match label {
                CompatibilityLabel::Safe => 0.0,
                CompatibilityLabel::Warning => 0.5,
                CompatibilityLabel::Breaking => 1.0,
            },
        };
        record.generation = Some(GeneratedRecordProvenance {
            oracle_jar_sha256: "test-jar".to_owned(),
            policy_packs_sha256: "test-packs".to_owned(),
            oracle_outcome: label.as_str().to_owned(),
            declared_mutation: mutation.to_owned(),
            mutation_variant: "test-variant".to_owned(),
            root_object_profile: None,
            oracle_stdout: format!("Schema compatibility: {}\n", label.as_str()),
            compatibility_mode: "BACKWARD".to_owned(),
            generator_version: "test".to_owned(),
            generation_seed: 1,
            label_source: "pinned-contract-cli-jar".to_owned(),
            oracle_invariant_rationale: Some("test oracle evidence".to_owned()),
            pair_fingerprint: format!("pair-{id}"),
        });
        record
    }

    #[test]
    fn audit_requires_open_and_closed_optional_field_provenance() {
        let mut record = generated_record(
            "optional-closed",
            "optional-closed-family",
            DatasetRole::Standard,
            "baseline",
            "optional_field_added",
            CompatibilityLabel::Safe,
        );
        record
            .generation
            .as_mut()
            .expect("generated record")
            .root_object_profile = Some("closed".to_owned());
        let report = PreparedDcgDataset::new("optional-profile-test", vec![record])
            .unwrap()
            .generalization_readiness();
        assert!(report.reasons.iter().any(|reason| {
            reason.contains("optional-field structural coverage has no `open` standard families")
        }));
        assert_eq!(
            report.optional_field_profile_coverage["closed"].standard_families,
            1
        );
        assert_eq!(
            report.optional_field_profile_coverage["open"].standard_families,
            0
        );
    }

    #[test]
    fn persists_explicit_optional_field_root_profile() {
        let mut record = generated_record(
            "optional-open",
            "optional-open-family",
            DatasetRole::Challenge,
            "baseline",
            "optional_field_added",
            CompatibilityLabel::Safe,
        );
        record
            .generation
            .as_mut()
            .expect("generated record")
            .root_object_profile = Some("open".to_owned());
        let dataset =
            PreparedDcgDataset::new("optional-profile-persistence", vec![record]).unwrap();
        let path = std::env::temp_dir().join(format!(
            "prepared-optional-profile-{}.json",
            std::process::id()
        ));
        dataset.save(&path).unwrap();
        let loaded = PreparedDcgDataset::load(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(
            loaded.records()[0]
                .generation
                .as_ref()
                .and_then(|generation| generation.root_object_profile.as_deref()),
            Some("open")
        );
    }

    #[test]
    fn counterfactual_audit_requires_isolated_multiclass_challenges() {
        let mut records = vec![
            generated_record(
                "s1",
                "standard-1",
                DatasetRole::Standard,
                "baseline",
                "add",
                CompatibilityLabel::Breaking,
            ),
            generated_record(
                "s2",
                "standard-2",
                DatasetRole::Standard,
                "strict",
                "add",
                CompatibilityLabel::Safe,
            ),
            generated_record(
                "s3",
                "standard-3",
                DatasetRole::Standard,
                "relaxed",
                "add",
                CompatibilityLabel::Warning,
            ),
        ];
        for (family, labels) in [
            (
                "challenge-1",
                [
                    CompatibilityLabel::Safe,
                    CompatibilityLabel::Warning,
                    CompatibilityLabel::Breaking,
                ],
            ),
            (
                "challenge-2",
                [
                    CompatibilityLabel::Breaking,
                    CompatibilityLabel::Safe,
                    CompatibilityLabel::Warning,
                ],
            ),
        ] {
            for (index, policy) in ["baseline", "strict", "relaxed"].iter().enumerate() {
                records.push(generated_record(
                    &format!("{family}-{policy}"),
                    family,
                    DatasetRole::Challenge,
                    policy,
                    "add",
                    labels[index],
                ));
            }
        }
        let dataset = PreparedDcgDataset::new("counterfactual-test", records).unwrap();
        let report = dataset.generalization_readiness();
        assert!(
            report.ready_for_generalization_benchmark,
            "{:?}",
            report.reasons
        );
        assert_eq!(report.complete_policy_counterfactual_pairs, 2);
        assert!(!report.challenge_family_leakage);
        assert!(!report.challenge_pair_leakage);
        let held_out = dataset
            .challenge_subset(&ChallengeProtocol::HeldOutPolicy {
                policy_pack: "baseline".to_owned(),
            })
            .unwrap();
        assert_eq!(held_out.len(), 2);
        let same_mutation = dataset
            .challenge_subset(&ChallengeProtocol::SameMutationDifferentPolicy)
            .unwrap();
        assert_eq!(same_mutation.len(), 6);
        let path = std::env::temp_dir().join(format!(
            "prepared-counterfactual-{}.json",
            std::process::id()
        ));
        dataset.save(&path).unwrap();
        let loaded = PreparedDcgDataset::load(&path).unwrap();
        assert_eq!(
            loaded.records()[0]
                .generation
                .as_ref()
                .and_then(|generation| generation.oracle_invariant_rationale.as_deref()),
            Some("test oracle evidence")
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn audit_blocks_a_breaking_label_without_an_oracle_decision() {
        let mut record = generated_record(
            "bad-break",
            "family-a",
            DatasetRole::Standard,
            "baseline",
            "field_removed",
            CompatibilityLabel::Breaking,
        );
        record
            .generation
            .as_mut()
            .expect("generated helper adds oracle provenance")
            .oracle_stdout = String::new();
        let dataset = PreparedDcgDataset::new("missing-decision", vec![record]).unwrap();
        let report = dataset.generalization_readiness();
        assert!(!report.ready_for_generalization_benchmark);
        assert!(
            report
                .reasons
                .iter()
                .any(|reason| reason.contains("no oracle decision stdout"))
        );
    }

    #[test]
    fn audit_requires_documented_oracle_invariant_mutations_for_exceptions() {
        let records = ["baseline", "strict", "relaxed"]
            .iter()
            .enumerate()
            .map(|(index, policy)| {
                generated_record(
                    &format!("invariant-{index}"),
                    "challenge-invariant",
                    DatasetRole::Challenge,
                    policy,
                    "always-breaks",
                    CompatibilityLabel::Breaking,
                )
            })
            .collect();
        let dataset = PreparedDcgDataset::new("invariant-test", records).unwrap();
        let promotions = OracleInvariantPromotionManifest {
            format_version: crate::models::ORACLE_PROMOTION_FORMAT_VERSION.to_owned(),
            promotions: ["baseline", "strict", "relaxed"]
                .iter()
                .map(
                    |policy_pack| crate::models::QualifiedOracleInvariantPromotion {
                        mutation_family: "always-breaks".to_owned(),
                        compatibility_mode: OracleCompatibilityMode::Backward,
                        policy_pack: (*policy_pack).to_owned(),
                        observed_label: "breaking".to_owned(),
                        oracle_jar_sha256: "test-jar".to_owned(),
                        policy_packs_sha256: "test-packs".to_owned(),
                        independent_families: 50,
                        oracle_checks: 150,
                        rationale: "reviewed pinned JAR evidence".to_owned(),
                    },
                )
                .collect(),
        };
        let report = dataset.generalization_readiness_with_promotions(&promotions);
        assert_eq!(
            report.observed_oracle_invariant_mutations,
            ["always-breaks"]
        );
        assert!(report.shortcut_risks.is_empty());
        assert!(!report.ready_for_generalization_benchmark);
        let mut stale_promotions = promotions.clone();
        stale_promotions.promotions[0].oracle_jar_sha256 = "changed-jar".to_owned();
        let stale = dataset.generalization_readiness_with_promotions(&stale_promotions);
        assert!(stale.observed_oracle_invariant_mutations.is_empty());
        assert!(!stale.shortcut_risks.is_empty());
    }
}
