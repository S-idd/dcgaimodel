//! Offline generation of oracle-labelled portable training records.

use super::{
    GeneratedPair, GeneratorError, MutationCandidate, MutationKind, OracleOutcome, OracleRun,
    PinnedOracle, candidates_for_schema,
};
use crate::features::{
    ApprovedPolicyContexts, DCG_FEATURE_V5_VERSION, DCG_FEATURE_V6_VERSION,
    SchemaChangeFeatureV6Extractor,
};
use crate::models::{
    CompatibilityLabel, ContractLabels, DatasetRole, GeneratedRecordProvenance,
    OracleCompatibilityMode, OracleInvariantPromotionManifest, PreparedDcgDataset,
    PreparedDcgRecord,
};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

/// Minimum corpus conditions used to drive a generation run. The executable
/// oracle remains the only label authority; these values only decide whether a
/// returned label is useful enough to keep searching for more families.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoverageTarget {
    /// Minimum retained records before a run may stop successfully.
    pub min_records: usize,
    /// Minimum non-challenge families available for train/validation/test.
    pub min_standard_families: usize,
    /// Minimum complete families held back for challenges.
    pub min_challenge_families: usize,
    /// Minimum family/mutation pairs evaluated under all configured policies.
    pub min_complete_policy_counterfactual_pairs: usize,
}

/// Minimum independent-family coverage for both label-free root-object
/// profiles of optional-field additions. This is a corpus-design gate, never
/// a source of labels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionalProfileCoverageTarget {
    pub min_standard_families_per_profile: usize,
    pub min_challenge_families_per_profile: usize,
}

impl OptionalProfileCoverageTarget {
    pub fn new(
        min_standard_families_per_profile: usize,
        min_challenge_families_per_profile: usize,
    ) -> Result<Self, GeneratorError> {
        if min_standard_families_per_profile == 0 || min_challenge_families_per_profile == 0 {
            return Err(GeneratorError::InvalidInput {
                field: "optional_profile_coverage_target",
            });
        }
        Ok(Self {
            min_standard_families_per_profile,
            min_challenge_families_per_profile,
        })
    }
}

impl CoverageTarget {
    /// Creates a bounded, family-first coverage target.
    pub fn new(
        min_records: usize,
        min_standard_families: usize,
        min_challenge_families: usize,
        min_complete_policy_counterfactual_pairs: usize,
    ) -> Result<Self, GeneratorError> {
        if min_records == 0
            || min_standard_families < 3
            || min_challenge_families == 0
            || min_complete_policy_counterfactual_pairs == 0
        {
            return Err(GeneratorError::InvalidInput {
                field: "coverage_target",
            });
        }
        Ok(Self {
            min_records,
            min_standard_families,
            min_challenge_families,
            min_complete_policy_counterfactual_pairs,
        })
    }
}

/// One source JSON Schema, with no compatibility claim attached to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedSchema {
    /// Stable source-provided identifier. All of its generated variants share a split group.
    pub id: String,
    /// Portable source/dataset identifier, not a local path.
    pub source: String,
    /// Raw JSON Schema document.
    pub schema: String,
}

/// A JSONSchemaBench parquet input file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonSchemaBenchSource {
    /// Local parquet path used only during generation; never written into output records.
    pub path: PathBuf,
    /// Portable source name included in generated records.
    pub source: String,
}

/// Label-free structural source selection used for targeted corpus work.
///
/// These profiles inspect only the source schema's root shape. They neither
/// invoke the oracle nor infer a compatibility label, so they are safe to use
/// before conformance and generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceSamplingProfile {
    /// Preserve the existing broad, priority-ranked source selection.
    Broad,
    /// Select schemas whose root object permits additional properties.
    OptionalFieldOpen,
    /// Select schemas whose root object explicitly forbids additional properties.
    OptionalFieldClosed,
    /// Take an equal, label-free source budget from optional-field open and
    /// closed root objects. If either source stratum is scarce, the returned
    /// sample is short and the configured coverage gate remains unsatisfied.
    BalancedOptional,
}

impl SourceSamplingProfile {
    /// Parses the portable CLI value for a structural profile.
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "broad" => Self::Broad,
            "optional-open" => Self::OptionalFieldOpen,
            "optional-closed" => Self::OptionalFieldClosed,
            "balanced-optional" => Self::BalancedOptional,
            _ => return None,
        })
    }
}

/// Result of a label-preserving V5-to-V6 feature re-extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefeatureReport {
    /// Deterministic source sampling seed recovered from generated evidence.
    pub generation_seed: u64,
    /// Source schemas replayed to rebuild the persisted pairs.
    pub sampled_source_families: usize,
    /// Persisted records whose source pair fingerprint was verified.
    pub verified_records: usize,
}

impl JsonSchemaBenchSource {
    /// Reads a deterministic, broadly distributed sample from the known
    /// `filename,json_schema,unique_id` layout.
    ///
    /// Sampling prioritizes schemas capable of producing a JAR warning under
    /// at least one policy pack, then ranks IDs within each group using `seed`.
    /// It avoids coupling a run to parquet order and makes the rare WARNING
    /// class practical to balance without ever manufacturing a label.
    pub fn load_sampled(
        &self,
        max_records: usize,
        seed: u64,
    ) -> Result<Vec<SeedSchema>, GeneratorError> {
        self.load_sampled_with_profile(max_records, seed, SourceSamplingProfile::Broad)
    }

    /// Reads a deterministic source sample constrained only by a label-free
    /// root-object structural profile.
    pub fn load_sampled_with_profile(
        &self,
        max_records: usize,
        seed: u64,
        profile: SourceSamplingProfile,
    ) -> Result<Vec<SeedSchema>, GeneratorError> {
        if max_records == 0 {
            return Err(GeneratorError::InvalidInput {
                field: "max_records",
            });
        }
        let file = File::open(&self.path).map_err(|error| GeneratorError::Source {
            message: error.to_string(),
        })?;
        let reader = SerializedFileReader::new(file).map_err(|error| GeneratorError::Source {
            message: error.to_string(),
        })?;
        let columns = reader
            .metadata()
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .map(|column| column.path().string())
            .collect::<Vec<_>>();
        if columns.as_slice() != ["filename", "json_schema", "unique_id"] {
            return Err(GeneratorError::Source {
                message: format!(
                    "expected JSONSchemaBench columns [filename, json_schema, unique_id], got {columns:?}"
                ),
            });
        }
        let mut sampled = BTreeMap::new();
        let open_limit = max_records / 2;
        let closed_limit = max_records - open_limit;
        let mut balanced_open = BTreeMap::new();
        let mut balanced_closed = BTreeMap::new();
        for row in reader
            .get_row_iter(None)
            .map_err(|error| GeneratorError::Source {
                message: error.to_string(),
            })?
        {
            let row = row.map_err(|error| GeneratorError::Source {
                message: error.to_string(),
            })?;
            let schema = row.get_string(1).map_err(|error| GeneratorError::Source {
                message: format!("json_schema column: {error}"),
            })?;
            let id = row.get_string(2).map_err(|error| GeneratorError::Source {
                message: format!("unique_id column: {error}"),
            })?;
            if !id.trim().is_empty() && !schema.trim().is_empty() {
                // Broad and balanced production sampling retain the historical
                // label-free class-supporting ordering *within each* root
                // profile. Targeted one-profile runs stay neutral.
                let priority = if matches!(
                    profile,
                    SourceSamplingProfile::Broad | SourceSamplingProfile::BalancedOptional
                ) {
                    source_priority(schema)
                } else {
                    0
                };
                let rank = stable_seed_rank(id, seed);
                let key = (priority, rank, id.to_owned());
                let seed_schema = SeedSchema {
                    id: id.to_owned(),
                    source: self.source.clone(),
                    schema: schema.to_owned(),
                };
                if profile == SourceSamplingProfile::BalancedOptional {
                    match root_object_profile(schema) {
                        Some("open") => {
                            balanced_open.insert(key, seed_schema);
                            if balanced_open.len() > open_limit {
                                balanced_open.pop_last();
                            }
                        }
                        Some("closed") => {
                            balanced_closed.insert(key, seed_schema);
                            if balanced_closed.len() > closed_limit {
                                balanced_closed.pop_last();
                            }
                        }
                        _ => {}
                    }
                } else if source_matches_profile(schema, profile) {
                    sampled.insert(key, seed_schema);
                    if sampled.len() > max_records {
                        sampled.pop_last();
                    }
                }
            }
        }
        if profile == SourceSamplingProfile::BalancedOptional {
            let mut combined = balanced_open
                .into_values()
                .chain(balanced_closed.into_values())
                .collect::<Vec<_>>();
            combined.sort_by_key(|seed_schema| {
                (
                    stable_seed_rank(&seed_schema.id, seed),
                    seed_schema.id.clone(),
                )
            });
            return Ok(combined);
        }
        Ok(sampled.into_values().collect())
    }
}

fn source_matches_profile(schema: &str, profile: SourceSamplingProfile) -> bool {
    if profile == SourceSamplingProfile::Broad {
        return true;
    }
    let Some(root_profile) = root_object_profile(schema) else {
        return false;
    };
    matches!(
        (profile, root_profile),
        (SourceSamplingProfile::OptionalFieldOpen, "open")
            | (SourceSamplingProfile::OptionalFieldClosed, "closed")
            | (SourceSamplingProfile::BalancedOptional, _)
    )
}

/// Returns the source root's label-free object profile only when it has root
/// properties and can produce the optional-field mutation.
fn root_object_profile(schema: &str) -> Option<&'static str> {
    let Value::Object(root) = serde_json::from_str::<Value>(schema).ok()? else {
        return None;
    };
    root.get("properties")
        .and_then(Value::as_object)
        .filter(|properties| !properties.is_empty())?;
    if root.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
        Some("closed")
    } else {
        Some("open")
    }
}

/// Re-extracts V6 label-free features from a V5 artifact without invoking the
/// oracle or modifying its labels.
///
/// Every retained record is reconstructed from the sampled source schema and
/// its generated mutation. The canonical pair fingerprint must exactly match
/// the V5 oracle evidence before V6 features are accepted. This makes the
/// resulting artifact a feature-only migration, not a fresh labelling run.
pub fn refeature_v5_as_v6(
    dataset: &PreparedDcgDataset,
    source: &JsonSchemaBenchSource,
    max_source_schemas: usize,
    policy_packs_path: &Path,
    oracle_jar_path: &Path,
    output_dataset_version: impl Into<String>,
) -> Result<(PreparedDcgDataset, RefeatureReport), GeneratorError> {
    if dataset.feature_version() != DCG_FEATURE_V5_VERSION {
        return Err(GeneratorError::Prepared {
            message: format!(
                "V6 re-feature extraction requires {DCG_FEATURE_V5_VERSION}, got {}",
                dataset.feature_version()
            ),
        });
    }
    let records = dataset.records();
    let generation_seeds = records
        .iter()
        .filter_map(|record| {
            record
                .generation
                .as_ref()
                .map(|generation| generation.generation_seed)
        })
        .collect::<BTreeSet<_>>();
    let [generation_seed] = generation_seeds.into_iter().collect::<Vec<_>>()[..] else {
        return Err(GeneratorError::Prepared {
            message: "V6 re-feature extraction requires exactly one generation seed".to_owned(),
        });
    };
    let expected_jar_hashes = records
        .iter()
        .filter_map(|record| {
            record
                .generation
                .as_ref()
                .map(|generation| generation.oracle_jar_sha256.as_str())
        })
        .collect::<BTreeSet<_>>();
    let expected_policy_hashes = records
        .iter()
        .filter_map(|record| {
            record
                .generation
                .as_ref()
                .map(|generation| generation.policy_packs_sha256.as_str())
        })
        .collect::<BTreeSet<_>>();
    let Some(expected_jar_hash) = single_value(&expected_jar_hashes) else {
        return Err(GeneratorError::Prepared {
            message: "V6 re-feature extraction requires one oracle JAR identity".to_owned(),
        });
    };
    let Some(expected_policy_hash) = single_value(&expected_policy_hashes) else {
        return Err(GeneratorError::Prepared {
            message: "V6 re-feature extraction requires one policy-pack identity".to_owned(),
        });
    };
    let actual_jar_hash = file_sha256(oracle_jar_path)?;
    if actual_jar_hash != expected_jar_hash {
        return Err(GeneratorError::Prepared {
            message: format!(
                "oracle JAR hash does not match V5 evidence: expected {expected_jar_hash}, got {actual_jar_hash}"
            ),
        });
    }
    let actual_policy_hash = file_sha256(policy_packs_path)?;
    if actual_policy_hash != expected_policy_hash {
        return Err(GeneratorError::Prepared {
            message: format!(
                "policy-pack hash does not match V5 evidence: expected {expected_policy_hash}, got {actual_policy_hash}"
            ),
        });
    }
    if records.iter().any(|record| record.source != source.source) {
        return Err(GeneratorError::Prepared {
            message: format!(
                "input records do not all identify source `{}`",
                source.source
            ),
        });
    }

    let policies = records
        .iter()
        .map(|record| record.policy_pack.clone())
        .collect::<Vec<_>>();
    let policy_contexts =
        ApprovedPolicyContexts::load(policy_packs_path, &policies).map_err(|error| {
            GeneratorError::Prepared {
                message: error.to_string(),
            }
        })?;
    let sampled = source.load_sampled(max_source_schemas, generation_seed)?;
    let schemas = sampled
        .into_iter()
        .map(|seed| (seed.id.clone(), seed))
        .collect::<BTreeMap<_, _>>();
    let mut rebuilt = Vec::with_capacity(records.len());
    for record in records {
        let generation = record
            .generation
            .as_ref()
            .ok_or_else(|| GeneratorError::Prepared {
                message: format!("record {} has no generated-pair evidence", record.record_id),
            })?;
        let seed = schemas
            .get(&record.family_id)
            .ok_or_else(|| GeneratorError::Prepared {
                message: format!(
                    "source sample does not contain persisted family {}",
                    record.family_id
                ),
            })?;
        let candidate = candidates_for_schema(&seed.schema)
            .map_err(|error| GeneratorError::Prepared {
                message: format!("could not rebuild candidates for {}: {error}", seed.id),
            })?
            .into_iter()
            .find(|candidate| {
                candidate.kind.as_str() == generation.declared_mutation
                    && candidate.variant == generation.mutation_variant
            })
            .ok_or_else(|| GeneratorError::Prepared {
                message: format!(
                    "source family {} no longer reproduces {}:{}",
                    seed.id, generation.declared_mutation, generation.mutation_variant
                ),
            })?;
        let fingerprint = canonical_pair_fingerprint(
            &seed.schema,
            &candidate.candidate_schema,
            &record.policy_pack,
            &actual_jar_hash,
        )?;
        if fingerprint != generation.pair_fingerprint {
            return Err(GeneratorError::Prepared {
                message: format!(
                    "pair fingerprint mismatch for {}; refusing feature migration",
                    record.record_id
                ),
            });
        }
        let mut updated = record.clone();
        if let Some(updated_generation) = &mut updated.generation {
            updated_generation.root_object_profile =
                root_object_profile(&seed.schema).map(str::to_owned);
        }
        updated.feature_version = DCG_FEATURE_V6_VERSION.to_owned();
        updated.features = SchemaChangeFeatureV6Extractor::new()
            .extract(
                &seed.schema,
                &candidate.candidate_schema,
                policy_contexts.get(&record.policy_pack).map_err(|error| {
                    GeneratorError::Prepared {
                        message: error.to_string(),
                    }
                })?,
            )
            .map_err(|error| GeneratorError::Prepared {
                message: error.to_string(),
            })?;
        rebuilt.push(updated);
    }
    let verified_records = rebuilt.len();
    let migrated = PreparedDcgDataset::new(output_dataset_version, rebuilt).map_err(|error| {
        GeneratorError::Prepared {
            message: error.to_string(),
        }
    })?;
    Ok((
        migrated,
        RefeatureReport {
            generation_seed,
            sampled_source_families: schemas.len(),
            verified_records,
        },
    ))
}

/// Controls a reproducible offline generation batch.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorConfig {
    /// Portable version assigned to the output prepared-data artifact.
    pub dataset_version: String,
    /// Every proposed transition is labelled once under each named policy pack.
    pub policy_packs: Vec<String>,
    /// Compatibility direction delegated verbatim to the pinned JAR. The
    /// default preserves the historical BACKWARD corpus workflow.
    pub compatibility_mode: OracleCompatibilityMode,
    /// Maximum number of input schemas consumed from the source.
    pub max_seed_schemas: usize,
    /// Stable seed for distributed source sampling and downstream split replay.
    pub seed: u64,
    /// Maximum independent JAR processes used for the policy checks belonging
    /// to one candidate. Results are always consumed in declared policy order,
    /// so this changes throughput only, never retention or quota semantics.
    pub oracle_workers: usize,
    /// Optional maximum retained records for each accepted JAR outcome.
    pub max_records_per_outcome: Option<usize>,
    /// Optional cap independently applied to each `(policy, mutation, outcome)`
    /// stratum, preserving oracle-observed counterfactuals across policies.
    pub max_records_per_policy_mutation_outcome: Option<usize>,
    /// Fraction of whole source families reserved for challenge evaluation.
    pub challenge_family_ratio: f64,
    /// Optional goal-driven stopping condition. `max_seed_schemas` remains a
    /// hard source-family budget, so a scarce oracle outcome never creates an
    /// unbounded run.
    pub coverage_target: Option<CoverageTarget>,
    /// Optional profile-level requirement for the V6 optional-field
    /// structural challenge. Both root profiles must be represented in both
    /// standard and challenge families before coverage-driven generation may
    /// succeed.
    pub optional_profile_coverage_target: Option<OptionalProfileCoverageTarget>,
    /// Explicit, hash-pinned qualified promotions used only when evaluating a
    /// coverage target. They cannot label records; the JAR remains the label
    /// authority and stale identities simply fail to match dataset evidence.
    pub promotion_manifest: Option<OracleInvariantPromotionManifest>,
    /// Optional label-free candidate-family filter for a focused generation
    /// batch. The pinned oracle still determines every retained label.
    pub mutation_filter: BTreeSet<String>,
    /// Mutation families for which coverage mode keeps each deterministic
    /// structural variant independently within a source family.
    pub retain_variants_for_mutations: BTreeSet<String>,
}

impl GeneratorConfig {
    /// Validates the generation configuration before any oracle work occurs.
    pub fn new(
        dataset_version: impl Into<String>,
        policy_packs: Vec<String>,
        max_seed_schemas: usize,
    ) -> Result<Self, GeneratorError> {
        let dataset_version = dataset_version.into();
        if dataset_version.trim().is_empty() {
            return Err(GeneratorError::InvalidInput {
                field: "dataset_version",
            });
        }
        if policy_packs.is_empty() || policy_packs.iter().any(|pack| pack.trim().is_empty()) {
            return Err(GeneratorError::InvalidInput {
                field: "policy_packs",
            });
        }
        if max_seed_schemas == 0 {
            return Err(GeneratorError::InvalidInput {
                field: "max_seed_schemas",
            });
        }
        Ok(Self {
            dataset_version,
            policy_packs,
            compatibility_mode: OracleCompatibilityMode::Backward,
            max_seed_schemas,
            seed: 0,
            oracle_workers: 1,
            max_records_per_outcome: None,
            max_records_per_policy_mutation_outcome: None,
            challenge_family_ratio: 0.15,
            coverage_target: None,
            optional_profile_coverage_target: None,
            promotion_manifest: None,
            mutation_filter: BTreeSet::new(),
            retain_variants_for_mutations: BTreeSet::new(),
        })
    }

    /// Uses a deterministic seed when sampling broadly from the source corpus.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Selects the one compatibility direction represented by this corpus.
    /// Directions stay separate because they are different oracle tasks.
    pub fn with_compatibility_mode(mut self, mode: OracleCompatibilityMode) -> Self {
        self.compatibility_mode = mode;
        self
    }

    /// Uses a bounded number of isolated JAR processes for each candidate's
    /// policy checks. One is the serial, backward-compatible default.
    pub fn with_oracle_workers(mut self, oracle_workers: usize) -> Result<Self, GeneratorError> {
        if oracle_workers == 0 {
            return Err(GeneratorError::InvalidInput {
                field: "oracle_workers",
            });
        }
        self.oracle_workers = oracle_workers;
        Ok(self)
    }

    /// Retains no more than this many SAFE, WARNING, and BREAKING records.
    ///
    /// Oracle calls are still made for candidates after one class fills, so a
    /// scarce warning class can be found without Rust assigning an outcome.
    pub fn with_outcome_balance(
        mut self,
        max_records_per_outcome: usize,
    ) -> Result<Self, GeneratorError> {
        if max_records_per_outcome == 0 {
            return Err(GeneratorError::InvalidInput {
                field: "max_records_per_outcome",
            });
        }
        self.max_records_per_outcome = Some(max_records_per_outcome);
        Ok(self)
    }

    /// Retains a bounded number of oracle outcomes in every policy/mutation
    /// stratum instead of allowing a global class quota to erase policy
    /// counterfactuals. This never fabricates a label when a stratum is empty.
    pub fn with_policy_mutation_outcome_balance(
        mut self,
        max_records_per_stratum: usize,
    ) -> Result<Self, GeneratorError> {
        if max_records_per_stratum == 0 {
            return Err(GeneratorError::InvalidInput {
                field: "max_records_per_policy_mutation_outcome",
            });
        }
        self.max_records_per_policy_mutation_outcome = Some(max_records_per_stratum);
        Ok(self)
    }

    /// Reserves a deterministic fraction of complete source families for
    /// challenge evaluation. They are persisted but excluded from normal
    /// train/validation/test splitting.
    pub fn with_challenge_family_ratio(mut self, ratio: f64) -> Result<Self, GeneratorError> {
        if !ratio.is_finite() || !(0.0..1.0).contains(&ratio) {
            return Err(GeneratorError::InvalidInput {
                field: "challenge_family_ratio",
            });
        }
        self.challenge_family_ratio = ratio;
        Ok(self)
    }

    /// Enables readiness-driven generation within the configured source budget.
    pub fn with_coverage_target(mut self, target: CoverageTarget) -> Self {
        self.coverage_target = Some(target);
        self
    }

    /// Requires independent optional-field families from both open and closed
    /// root-object profiles in each evaluation role. The profile is read from
    /// the source schema before oracle invocation and never from a label.
    pub fn with_optional_profile_coverage_target(
        mut self,
        target: OptionalProfileCoverageTarget,
    ) -> Self {
        self.optional_profile_coverage_target = Some(target);
        self
    }

    /// Supplies a reviewed promotion manifest to the benchmark-readiness gate.
    pub fn with_promotion_manifest(mut self, manifest: OracleInvariantPromotionManifest) -> Self {
        self.promotion_manifest = Some(manifest);
        self
    }

    /// Limits candidate proposals to declared mutation families without ever
    /// assigning their labels in Rust.
    pub fn with_mutation_filter(
        mut self,
        mutations: impl IntoIterator<Item = String>,
    ) -> Result<Self, GeneratorError> {
        let mutations = mutations.into_iter().collect::<BTreeSet<_>>();
        if mutations
            .iter()
            .any(|mutation| MutationKind::parse(mutation).is_none())
        {
            return Err(GeneratorError::InvalidInput {
                field: "mutation_filter",
            });
        }
        self.mutation_filter = mutations;
        Ok(self)
    }

    /// Retains each requested mutation variant separately under a
    /// per-family coverage cap. This prevents a first candidate such as
    /// `optional-string-field` from crowding out integer, boolean, and array
    /// controls in a targeted corpus.
    pub fn with_variant_retention(
        mut self,
        mutations: impl IntoIterator<Item = String>,
    ) -> Result<Self, GeneratorError> {
        let mutations = mutations.into_iter().collect::<BTreeSet<_>>();
        if mutations
            .iter()
            .any(|mutation| MutationKind::parse(mutation).is_none())
        {
            return Err(GeneratorError::InvalidInput {
                field: "retain_variants_for_mutations",
            });
        }
        self.retain_variants_for_mutations = mutations;
        Ok(self)
    }
}

/// Counters for an auditable generator run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GeneratorReport {
    /// Source records read up to the requested limit.
    pub seeds_seen: usize,
    /// Candidate transitions proposed before oracle execution.
    pub candidates_proposed: usize,
    /// SAFE records accepted from clean JAR exit-0 output.
    pub oracle_safe: usize,
    /// WARNING records accepted from JAR exit-0 warning output.
    pub oracle_warning: usize,
    /// Records accepted with a breaking outcome.
    pub oracle_breaking: usize,
    /// Candidates rejected because the oracle returned exit code 2.
    pub oracle_rejected: usize,
    /// Accepted JAR results omitted after their requested outcome quota filled.
    pub outcome_quota_skipped: usize,
    /// Accepted oracle outcomes rejected because their canonical pair key was already retained.
    pub duplicates_rejected: usize,
    /// Seeds skipped because they were not JSON documents or had no supported mutation.
    pub seeds_skipped: usize,
    /// Accepted record counts indexed by `policy:mutation:oracle-outcome`.
    pub accepted_policy_mutation_outcomes: BTreeMap<String, usize>,
    /// True only when the requested coverage target and the persisted-dataset
    /// benchmark gate both passed before the source budget was exhausted.
    pub coverage_target_reached: Option<bool>,
    /// Concrete unmet conditions when a goal-driven batch ends without a
    /// benchmark-ready corpus.
    pub coverage_reasons: Vec<String>,
}

/// Builds a portable dataset by asking the executable oracle about every pair.
#[derive(Debug, Clone)]
pub struct TrainingDataGenerator {
    oracle: PinnedOracle,
}

impl TrainingDataGenerator {
    /// Creates a generator bound to a single pinned oracle executable.
    pub fn new(oracle: PinnedOracle) -> Self {
        Self { oracle }
    }

    /// Consumes in-memory seeds and returns only oracle-labelled records.
    ///
    /// `workspace` is staging-only. The output contains no paths to the source
    /// parquet, workspace, Java checkout, or policy-pack file.
    pub fn generate(
        &self,
        config: &GeneratorConfig,
        workspace: &Path,
        seeds: impl IntoIterator<Item = SeedSchema>,
    ) -> Result<(PreparedDcgDataset, GeneratorReport), GeneratorError> {
        let policy_contexts =
            ApprovedPolicyContexts::load(self.oracle.policy_packs_path(), &config.policy_packs)
                .map_err(|error| GeneratorError::Prepared {
                    message: error.to_string(),
                })?;
        let mut records = Vec::new();
        let mut retained_fingerprints = HashSet::new();
        let mut retained_family_strata = BTreeMap::<String, usize>::new();
        let mut report = GeneratorReport::default();
        'seeds: for seed in seeds.into_iter().take(config.max_seed_schemas) {
            report.seeds_seen += 1;
            if seed.id.trim().is_empty() || seed.source.trim().is_empty() {
                report.seeds_skipped += 1;
                continue;
            }
            let mut candidates = match candidates_for_schema(&seed.schema) {
                Ok(candidates) if !candidates.is_empty() => candidates,
                _ => {
                    report.seeds_skipped += 1;
                    continue;
                }
            };
            if !config.mutation_filter.is_empty() {
                candidates
                    .retain(|candidate| config.mutation_filter.contains(candidate.kind.as_str()));
                if candidates.is_empty() {
                    report.seeds_skipped += 1;
                    continue;
                }
            }
            // Spend oracle calls on mutation families with the least retained
            // evidence first. The oracle still determines every outcome; this
            // only controls proposal order within an independent family.
            candidates.sort_by_key(|candidate| {
                report
                    .accepted_policy_mutation_outcomes
                    .iter()
                    .filter(|(stratum, _)| {
                        stratum
                            .split(':')
                            .nth(1)
                            .is_some_and(|mutation| mutation == candidate.kind.as_str())
                    })
                    .map(|(_, count)| *count)
                    .sum::<usize>()
            });
            let dataset_role =
                if reserve_challenge_family(&seed.id, config.seed, config.challenge_family_ratio) {
                    DatasetRole::Challenge
                } else {
                    DatasetRole::Standard
                };
            let root_object_profile = root_object_profile(&seed.schema).map(str::to_owned);
            for candidate in candidates {
                let policy_runs = self.check_candidate_policies(
                    workspace,
                    &seed,
                    &candidate,
                    &config.policy_packs,
                    config.oracle_workers,
                    config.compatibility_mode,
                )?;
                // `check_candidate_policies` may execute JAR processes in
                // parallel, but returns these in declared policy order. Keep
                // all quota, duplicate, feature, and record work here on one
                // thread so a worker count never changes corpus semantics.
                for (policy_pack, run) in policy_runs {
                    report.candidates_proposed += 1;
                    if run.outcome == OracleOutcome::Rejected {
                        report.oracle_rejected += 1;
                        continue;
                    }
                    let pair_fingerprint = canonical_pair_fingerprint_for_mode(
                        &seed.schema,
                        &candidate.candidate_schema,
                        &policy_pack,
                        self.oracle.jar_sha256(),
                        config.compatibility_mode,
                    )?;
                    if retained_fingerprints.contains(&pair_fingerprint) {
                        report.duplicates_rejected += 1;
                        continue;
                    }
                    let stratum = format!(
                        "{}:{}:{}",
                        policy_pack,
                        candidate.kind.as_str(),
                        run.outcome.as_str()
                    );
                    let family_stratum = if config
                        .retain_variants_for_mutations
                        .contains(candidate.kind.as_str())
                    {
                        format!("{}:{stratum}:{}", seed.id, candidate.variant)
                    } else {
                        format!("{}:{stratum}", seed.id)
                    };
                    if !retain_policy_mutation_outcome(
                        config,
                        &report,
                        &retained_family_strata,
                        &stratum,
                        &family_stratum,
                    ) {
                        report.outcome_quota_skipped += 1;
                        continue;
                    }
                    let (labels, outcome_name, compatibility_label) = match run.outcome {
                        OracleOutcome::Safe => {
                            if !retain_outcome(config, &report, OracleOutcome::Safe) {
                                report.outcome_quota_skipped += 1;
                                continue;
                            }
                            report.oracle_safe += 1;
                            (
                                labels_for(OracleOutcome::Safe),
                                OracleOutcome::Safe.as_str(),
                                CompatibilityLabel::Safe,
                            )
                        }
                        OracleOutcome::Warning => {
                            if !retain_outcome(config, &report, OracleOutcome::Warning) {
                                report.outcome_quota_skipped += 1;
                                continue;
                            }
                            report.oracle_warning += 1;
                            (
                                labels_for(OracleOutcome::Warning),
                                OracleOutcome::Warning.as_str(),
                                CompatibilityLabel::Warning,
                            )
                        }
                        OracleOutcome::Breaking => {
                            if !retain_outcome(config, &report, OracleOutcome::Breaking) {
                                report.outcome_quota_skipped += 1;
                                continue;
                            }
                            report.oracle_breaking += 1;
                            (
                                labels_for(OracleOutcome::Breaking),
                                OracleOutcome::Breaking.as_str(),
                                CompatibilityLabel::Breaking,
                            )
                        }
                        OracleOutcome::Rejected => {
                            unreachable!("rejected outcomes are handled above")
                        }
                    };
                    // A quota-skipped outcome must not reserve the identity:
                    // another independent source candidate may be the record
                    // that actually fills an eligible counterfactual stratum.
                    retained_fingerprints.insert(pair_fingerprint.clone());
                    records.push(PreparedDcgRecord {
                        record_id: format!(
                            "{}:{}:{}:{}",
                            seed.id,
                            candidate.kind.as_str(),
                            candidate.variant,
                            policy_pack
                        ),
                        source: seed.source.clone(),
                        family_id: seed.id.clone(),
                        split_group_id: seed.id.clone(),
                        dataset_role,
                        contract_id: seed.id.clone(),
                        old_version: "1.0.0".to_owned(),
                        new_version: "1.0.1".to_owned(),
                        policy_pack: policy_pack.clone(),
                        feature_version: DCG_FEATURE_V6_VERSION.to_owned(),
                        features: SchemaChangeFeatureV6Extractor::new()
                            .extract(
                                &seed.schema,
                                &candidate.candidate_schema,
                                policy_contexts.get(&policy_pack).map_err(|error| {
                                    GeneratorError::Prepared {
                                        message: error.to_string(),
                                    }
                                })?,
                            )
                            .map_err(|error| GeneratorError::Prepared {
                                message: error.to_string(),
                            })?,
                        labels,
                        compatibility_label: Some(compatibility_label),
                        generation: Some(GeneratedRecordProvenance {
                            oracle_jar_sha256: self.oracle.jar_sha256().to_owned(),
                            policy_packs_sha256: self.oracle.policy_packs_sha256().to_owned(),
                            oracle_outcome: outcome_name.to_owned(),
                            declared_mutation: candidate.kind.as_str().to_owned(),
                            mutation_variant: candidate.variant.clone(),
                            root_object_profile: root_object_profile.clone(),
                            oracle_stdout: run.stdout,
                            compatibility_mode: config.compatibility_mode.as_str().to_owned(),
                            generator_version: env!("CARGO_PKG_VERSION").to_owned(),
                            generation_seed: config.seed,
                            label_source: "pinned-contract-cli-jar".to_owned(),
                            oracle_invariant_rationale: None,
                            pair_fingerprint,
                        }),
                    });
                    *report
                        .accepted_policy_mutation_outcomes
                        .entry(stratum)
                        .or_default() += 1;
                    *retained_family_strata.entry(family_stratum).or_default() += 1;
                }
            }
            if let Some(target) = &config.coverage_target {
                // Coverage is defined over the persisted artifact. Move any
                // missing variable-mutation outcomes into whole-family
                // challenge holdout *before* deciding that the target has
                // been met. Otherwise the final reservation can reduce the
                // standard-family count after a seemingly successful check.
                reserve_missing_variable_mutation_outcomes(&mut records);
                let (ready, reasons) = coverage_status(&records, config, target);
                report.coverage_target_reached = Some(ready);
                report.coverage_reasons = reasons;
                if ready {
                    break 'seeds;
                }
            } else if outcome_balance_reached(config, &report) {
                break 'seeds;
            }
        }
        if let Some(target) = &config.coverage_target
            && report.coverage_target_reached != Some(true)
        {
            reserve_missing_variable_mutation_outcomes(&mut records);
            let (ready, reasons) = coverage_status(&records, config, target);
            report.coverage_target_reached = Some(ready);
            report.coverage_reasons = reasons;
        }
        if config.coverage_target.is_none() {
            reserve_missing_variable_mutation_outcomes(&mut records);
        }
        let dataset =
            PreparedDcgDataset::new(config.dataset_version.clone(), records).map_err(|error| {
                GeneratorError::Prepared {
                    message: error.to_string(),
                }
            })?;
        Ok((dataset, report))
    }

    /// Executes one candidate under all declared policies. Individual JAR
    /// processes are isolated by [`PinnedOracle`]'s unique staging directory;
    /// their results are joined in the original policy order before any
    /// mutable generator state observes them.
    fn check_candidate_policies(
        &self,
        workspace: &Path,
        seed: &SeedSchema,
        candidate: &MutationCandidate,
        policy_packs: &[String],
        oracle_workers: usize,
        mode: OracleCompatibilityMode,
    ) -> Result<Vec<(String, OracleRun)>, GeneratorError> {
        if oracle_workers == 1 {
            return policy_packs
                .iter()
                .map(|policy_pack| {
                    self.oracle
                        .check(
                            workspace,
                            &GeneratedPair {
                                contract_id: seed.id.clone(),
                                policy_pack: policy_pack.clone(),
                                base_schema: seed.schema.clone(),
                                candidate_schema: candidate.candidate_schema.clone(),
                            },
                            mode,
                        )
                        .map(|run| (policy_pack.clone(), run))
                })
                .collect();
        }

        let worker_count = oracle_workers.min(policy_packs.len());
        let mut ordered_runs = Vec::with_capacity(policy_packs.len());
        for policy_batch in policy_packs.chunks(worker_count) {
            let batch_runs = std::thread::scope(|scope| {
                let handles = policy_batch
                    .iter()
                    .cloned()
                    .map(|policy_pack| {
                        let oracle = self.oracle.clone();
                        let workspace = workspace.to_path_buf();
                        let contract_id = seed.id.clone();
                        let base_schema = seed.schema.clone();
                        let candidate_schema = candidate.candidate_schema.clone();
                        let worker_policy_pack = policy_pack.clone();
                        let handle = scope.spawn(move || {
                            oracle.check(
                                &workspace,
                                &GeneratedPair {
                                    contract_id,
                                    policy_pack: worker_policy_pack,
                                    base_schema,
                                    candidate_schema,
                                },
                                mode,
                            )
                        });
                        (policy_pack, handle)
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|(policy_pack, handle)| {
                        let run = handle.join().map_err(|_| GeneratorError::Io {
                            message: format!(
                                "oracle worker panicked while checking policy pack `{policy_pack}`"
                            ),
                        })??;
                        Ok((policy_pack, run))
                    })
                    .collect::<Result<Vec<_>, GeneratorError>>()
            });
            ordered_runs.extend(batch_runs?);
        }
        Ok(ordered_runs)
    }
}

fn coverage_status(
    records: &[PreparedDcgRecord],
    config: &GeneratorConfig,
    target: &CoverageTarget,
) -> (bool, Vec<String>) {
    let mut reasons = Vec::new();
    if records.len() < target.min_records {
        reasons.push(format!(
            "records={} is below target={}",
            records.len(),
            target.min_records
        ));
    }
    let standard_families = records
        .iter()
        .filter(|record| record.dataset_role == DatasetRole::Standard)
        .map(|record| &record.family_id)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if standard_families < target.min_standard_families {
        reasons.push(format!(
            "standard_families={standard_families} is below target={}",
            target.min_standard_families
        ));
    }
    let challenge_families = records
        .iter()
        .filter(|record| record.dataset_role == DatasetRole::Challenge)
        .map(|record| &record.family_id)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    if challenge_families < target.min_challenge_families {
        reasons.push(format!(
            "challenge_families={challenge_families} is below target={}",
            target.min_challenge_families
        ));
    }
    if let Some(profile_target) = &config.optional_profile_coverage_target {
        for profile in ["open", "closed"] {
            let standard_profile_families = records
                .iter()
                .filter(|record| {
                    record.dataset_role == DatasetRole::Standard
                        && record.generation.as_ref().is_some_and(|generation| {
                            generation.declared_mutation == "optional_field_added"
                                && generation.root_object_profile.as_deref() == Some(profile)
                        })
                })
                .map(|record| &record.family_id)
                .collect::<BTreeSet<_>>()
                .len();
            if standard_profile_families < profile_target.min_standard_families_per_profile {
                reasons.push(format!(
                    "optional_{profile}_standard_families={standard_profile_families} is below target={}",
                    profile_target.min_standard_families_per_profile
                ));
            }
            let challenge_profile_families = records
                .iter()
                .filter(|record| {
                    record.dataset_role == DatasetRole::Challenge
                        && record.generation.as_ref().is_some_and(|generation| {
                            generation.declared_mutation == "optional_field_added"
                                && generation.root_object_profile.as_deref() == Some(profile)
                        })
                })
                .map(|record| &record.family_id)
                .collect::<BTreeSet<_>>()
                .len();
            if challenge_profile_families < profile_target.min_challenge_families_per_profile {
                reasons.push(format!(
                    "optional_{profile}_challenge_families={challenge_profile_families} is below target={}",
                    profile_target.min_challenge_families_per_profile
                ));
            }
        }
    }
    if records.is_empty() {
        return (false, reasons);
    }
    let dataset = match PreparedDcgDataset::new(config.dataset_version.clone(), records.to_vec()) {
        Ok(dataset) => dataset,
        Err(error) => {
            return (
                false,
                vec![format!("prepared dataset validation failed: {error}")],
            );
        }
    };
    let readiness = config.promotion_manifest.as_ref().map_or_else(
        || dataset.generalization_readiness(),
        |manifest| dataset.generalization_readiness_with_promotions(manifest),
    );
    if readiness.complete_policy_counterfactual_pairs
        < target.min_complete_policy_counterfactual_pairs
    {
        reasons.push(format!(
            "complete_policy_counterfactual_pairs={} is below target={}",
            readiness.complete_policy_counterfactual_pairs,
            target.min_complete_policy_counterfactual_pairs
        ));
    }
    if !readiness.ready_for_generalization_benchmark {
        reasons.extend(
            readiness
                .reasons
                .into_iter()
                .map(|reason| format!("benchmark gate: {reason}")),
        );
    }
    (reasons.is_empty(), reasons)
}

/// Ensures each mutation with more than one oracle-observed label has those
/// labels represented in family-isolated challenge data. Roles are changed
/// only after the JAR has labelled every retained record, and a whole source
/// family always moves together.
fn reserve_missing_variable_mutation_outcomes(records: &mut [PreparedDcgRecord]) {
    let mut all_labels = BTreeMap::<String, BTreeSet<String>>::new();
    let mut challenge_labels = BTreeMap::<String, BTreeSet<String>>::new();
    for record in records.iter() {
        let Some(generation) = &record.generation else {
            continue;
        };
        let mutation = generation.declared_mutation.clone();
        let label = record.compatibility_label.map_or_else(
            || generation.oracle_outcome.clone(),
            |label| label.as_str().to_owned(),
        );
        all_labels
            .entry(mutation.clone())
            .or_default()
            .insert(label.clone());
        if record.dataset_role == DatasetRole::Challenge {
            challenge_labels.entry(mutation).or_default().insert(label);
        }
    }
    let mut selected_families = BTreeSet::new();
    for (mutation, labels) in all_labels {
        if labels.len() < 2 {
            continue;
        }
        for label in labels {
            if challenge_labels
                .get(&mutation)
                .is_some_and(|present| present.contains(&label))
            {
                continue;
            }
            if let Some(family) = records.iter().find_map(|record| {
                let generation = record.generation.as_ref()?;
                let record_label = record.compatibility_label.map_or_else(
                    || generation.oracle_outcome.as_str(),
                    |value| value.as_str(),
                );
                (record.dataset_role == DatasetRole::Standard
                    && generation.declared_mutation == mutation
                    && record_label == label)
                    .then(|| record.family_id.clone())
            }) {
                selected_families.insert(family);
            }
        }
    }
    for record in records {
        if selected_families.contains(&record.family_id) {
            record.dataset_role = DatasetRole::Challenge;
        }
    }
}

fn reserve_challenge_family(id: &str, seed: u64, ratio: f64) -> bool {
    if ratio == 0.0 {
        return false;
    }
    // Source sampling also uses a simple stable rank. Use a distinct SHA-256
    // domain here, both to separate the decisions and to avoid clustering
    // nearby source IDs into the same reservation outcome.
    challenge_rank(id, seed) as f64 / (u64::MAX as f64) < ratio
}

fn challenge_rank(id: &str, seed: u64) -> u64 {
    let digest = Sha256::digest(format!("dcg-challenge-v1:{seed}:{id}").as_bytes());
    u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    )
}

/// Returns a reproducible identity for a proposed compatibility transition.
/// Object keys are recursively sorted; array order is intentionally retained
/// because JSON Schema arrays such as `oneOf` can be semantically ordered.
pub fn canonical_pair_fingerprint(
    base_schema: &str,
    candidate_schema: &str,
    policy_pack: &str,
    oracle_jar_sha256: &str,
) -> Result<String, GeneratorError> {
    canonical_pair_fingerprint_for_mode(
        base_schema,
        candidate_schema,
        policy_pack,
        oracle_jar_sha256,
        OracleCompatibilityMode::Backward,
    )
}

/// Direction-aware identity for a proposed compatibility transition. The
/// legacy wrapper remains BACKWARD-only so V9 and external-audit hashes stay
/// byte-for-byte stable.
pub fn canonical_pair_fingerprint_for_mode(
    base_schema: &str,
    candidate_schema: &str,
    policy_pack: &str,
    oracle_jar_sha256: &str,
    mode: OracleCompatibilityMode,
) -> Result<String, GeneratorError> {
    let base: Value =
        serde_json::from_str(base_schema).map_err(|error| GeneratorError::Prepared {
            message: format!("base schema is not canonicalizable JSON: {error}"),
        })?;
    let candidate: Value =
        serde_json::from_str(candidate_schema).map_err(|error| GeneratorError::Prepared {
            message: format!("candidate schema is not canonicalizable JSON: {error}"),
        })?;
    let payload = format!(
        "base={}\ncandidate={}\npolicy={}\nmode={}\noracle={}",
        canonical_json(&base),
        canonical_json(&candidate),
        policy_pack,
        mode.as_str(),
        oracle_jar_sha256,
    );
    Ok(format!("{:x}", Sha256::digest(payload.as_bytes())))
}

fn file_sha256(path: &Path) -> Result<String, GeneratorError> {
    let contents = std::fs::read(path).map_err(|error| GeneratorError::Io {
        message: error.to_string(),
    })?;
    Ok(format!("{:x}", Sha256::digest(contents)))
}

fn single_value<'a>(values: &'a BTreeSet<&'a str>) -> Option<&'a str> {
    (values.len() == 1)
        .then(|| values.first().copied())
        .flatten()
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_by_key(|(key, _)| *key);
            format!(
                "{{{}}}",
                entries
                    .into_iter()
                    .map(|(key, value)| format!(
                        "{}:{}",
                        serde_json::to_string(key).expect("string keys serialize"),
                        canonical_json(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        _ => serde_json::to_string(value).expect("JSON scalar serializes"),
    }
}

fn retain_outcome(
    config: &GeneratorConfig,
    report: &GeneratorReport,
    outcome: OracleOutcome,
) -> bool {
    let Some(limit) = config.max_records_per_outcome else {
        return true;
    };
    match outcome {
        OracleOutcome::Safe => report.oracle_safe < limit,
        OracleOutcome::Warning => report.oracle_warning < limit,
        OracleOutcome::Breaking => report.oracle_breaking < limit,
        OracleOutcome::Rejected => false,
    }
}

fn retain_policy_mutation_outcome(
    config: &GeneratorConfig,
    report: &GeneratorReport,
    family_strata: &BTreeMap<String, usize>,
    stratum: &str,
    family_stratum: &str,
) -> bool {
    config
        .max_records_per_policy_mutation_outcome
        .is_none_or(|limit| {
            if config.coverage_target.is_some() {
                // A global cap is appropriate for a tiny balanced smoke
                // artifact. A coverage-driven corpus needs many independent
                // families in each useful stratum, so bound repeated variants
                // only within one source family instead.
                family_strata.get(family_stratum).copied().unwrap_or(0) < limit
            } else {
                report
                    .accepted_policy_mutation_outcomes
                    .get(stratum)
                    .copied()
                    .unwrap_or(0)
                    < limit
            }
        })
}

fn outcome_balance_reached(config: &GeneratorConfig, report: &GeneratorReport) -> bool {
    config.max_records_per_outcome.is_some_and(|limit| {
        report.oracle_safe >= limit
            && report.oracle_warning >= limit
            && report.oracle_breaking >= limit
    })
}

fn labels_for(outcome: OracleOutcome) -> ContractLabels {
    match outcome {
        OracleOutcome::Safe => ContractLabels {
            breaking_change: 0.0,
            incompatible: 0.0,
            risk_score: 0.0,
        },
        OracleOutcome::Warning => ContractLabels {
            breaking_change: 0.0,
            incompatible: 0.0,
            // This is the only non-binary confidence signal the JAR exposes:
            // a compatible decision with warnings. Preserve it exactly as a
            // mid-risk category instead of pretending it is safely clean.
            risk_score: 0.5,
        },
        OracleOutcome::Breaking => ContractLabels {
            breaking_change: 1.0,
            incompatible: 1.0,
            risk_score: 1.0,
        },
        OracleOutcome::Rejected => unreachable!("rejected oracle outcomes are never retained"),
    }
}

fn stable_seed_rank(id: &str, seed: u64) -> u64 {
    id.bytes().fold(seed ^ 0xcbf29ce484222325, |hash, byte| {
        (hash ^ byte as u64).wrapping_mul(0x100000001b3)
    })
}

fn source_priority(schema: &str) -> u8 {
    match candidates_for_schema(schema) {
        Ok(candidates)
            if candidates
                .iter()
                .any(|candidate| candidate.kind == MutationKind::EnumValueAdded) =>
        {
            0
        }
        Ok(candidates) if !candidates.is_empty() => 1,
        _ => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_a_nonempty_batch_configuration() {
        assert!(GeneratorConfig::new("oracle-v1", vec!["baseline".to_owned()], 10).is_ok());
        assert!(GeneratorConfig::new("", vec!["baseline".to_owned()], 10).is_err());
        assert!(CoverageTarget::new(100, 3, 1, 1).is_ok());
        assert!(CoverageTarget::new(100, 2, 1, 1).is_err());
        assert!(OptionalProfileCoverageTarget::new(1, 1).is_ok());
        assert!(OptionalProfileCoverageTarget::new(0, 1).is_err());
    }

    #[test]
    fn validates_oracle_worker_configuration() {
        let config = GeneratorConfig::new("oracle-v1", vec!["baseline".to_owned()], 10)
            .unwrap()
            .with_oracle_workers(8)
            .unwrap();
        assert_eq!(config.oracle_workers, 8);
        assert!(
            GeneratorConfig::new("oracle-v1", vec!["baseline".to_owned()], 10)
                .unwrap()
                .with_oracle_workers(0)
                .is_err()
        );
    }

    #[test]
    fn validates_targeted_optional_sampling_profiles_without_labels() {
        let open = r#"{"type":"object","properties":{"name":{"type":"string"}}}"#;
        let closed = r#"{"type":"object","additionalProperties":false,"properties":{"name":{"type":"string"}}}"#;
        let no_properties = r#"{"type":"object","additionalProperties":false}"#;
        assert!(source_matches_profile(
            open,
            SourceSamplingProfile::OptionalFieldOpen
        ));
        assert!(!source_matches_profile(
            open,
            SourceSamplingProfile::OptionalFieldClosed
        ));
        assert!(source_matches_profile(
            closed,
            SourceSamplingProfile::OptionalFieldClosed
        ));
        assert!(!source_matches_profile(
            closed,
            SourceSamplingProfile::OptionalFieldOpen
        ));
        assert!(!source_matches_profile(
            no_properties,
            SourceSamplingProfile::OptionalFieldClosed
        ));
        assert!(source_matches_profile(
            open,
            SourceSamplingProfile::BalancedOptional
        ));
        assert!(source_matches_profile(
            closed,
            SourceSamplingProfile::BalancedOptional
        ));
        assert_eq!(root_object_profile(open), Some("open"));
        assert_eq!(root_object_profile(closed), Some("closed"));
    }

    #[test]
    fn retains_requested_variants_independently_in_coverage_mode() {
        let config = GeneratorConfig::new("oracle-v1", vec!["baseline".to_owned()], 10)
            .unwrap()
            .with_policy_mutation_outcome_balance(1)
            .unwrap()
            .with_coverage_target(CoverageTarget::new(10, 3, 1, 1).unwrap())
            .with_mutation_filter(vec!["optional_field_added".to_owned()])
            .unwrap()
            .with_variant_retention(vec!["optional_field_added".to_owned()])
            .unwrap();
        assert!(config.mutation_filter.contains("optional_field_added"));
        assert!(
            config
                .retain_variants_for_mutations
                .contains("optional_field_added")
        );
        let families = BTreeMap::from([(
            "family-a:baseline:optional_field_added:safe:optional-string-field".to_owned(),
            1,
        )]);
        assert!(retain_policy_mutation_outcome(
            &config,
            &GeneratorReport::default(),
            &families,
            "baseline:optional_field_added:safe",
            "family-a:baseline:optional_field_added:safe:optional-integer-field",
        ));
    }

    #[test]
    fn outcome_labels_preserve_the_jar_outcome_categories() {
        assert_eq!(labels_for(OracleOutcome::Safe).risk_score, 0.0);
        assert_eq!(labels_for(OracleOutcome::Warning).risk_score, 0.5);
        assert_eq!(labels_for(OracleOutcome::Breaking).risk_score, 1.0);
        assert_eq!(labels_for(OracleOutcome::Warning).breaking_change, 0.0);
    }

    #[test]
    fn balance_quota_is_independent_for_each_accepted_outcome() {
        let config = GeneratorConfig::new("oracle-v1", vec!["baseline".to_owned()], 10)
            .unwrap()
            .with_outcome_balance(2)
            .unwrap();
        let report = GeneratorReport {
            oracle_safe: 2,
            oracle_warning: 1,
            oracle_breaking: 2,
            ..GeneratorReport::default()
        };
        assert!(!retain_outcome(&config, &report, OracleOutcome::Safe));
        assert!(retain_outcome(&config, &report, OracleOutcome::Warning));
        assert!(!retain_outcome(&config, &report, OracleOutcome::Breaking));
        assert!(!outcome_balance_reached(&config, &report));
    }

    #[test]
    fn coverage_mode_keeps_the_stratum_cap_per_independent_family() {
        let config = GeneratorConfig::new("oracle-v1", vec!["baseline".to_owned()], 10)
            .unwrap()
            .with_policy_mutation_outcome_balance(1)
            .unwrap()
            .with_coverage_target(CoverageTarget::new(10, 3, 1, 1).unwrap());
        let mut report = GeneratorReport::default();
        report
            .accepted_policy_mutation_outcomes
            .insert("baseline:field_removed:breaking".to_owned(), 10);
        let mut families = BTreeMap::new();
        families.insert("family-a:baseline:field_removed:breaking".to_owned(), 1);
        assert!(!retain_policy_mutation_outcome(
            &config,
            &report,
            &families,
            "baseline:field_removed:breaking",
            "family-a:baseline:field_removed:breaking",
        ));
        assert!(retain_policy_mutation_outcome(
            &config,
            &report,
            &families,
            "baseline:field_removed:breaking",
            "family-b:baseline:field_removed:breaking",
        ));
    }

    #[test]
    fn source_sampling_rank_is_reproducible_and_seed_sensitive() {
        assert_eq!(
            stable_seed_rank("family-a", 7),
            stable_seed_rank("family-a", 7)
        );
        assert_ne!(
            stable_seed_rank("family-a", 7),
            stable_seed_rank("family-a", 8)
        );
    }

    #[test]
    fn source_sampling_prioritizes_warning_capable_schemas_without_labelling_them() {
        let enum_schema =
            r#"{"type":"object","properties":{"status":{"type":"string","enum":["a"]}}}"#;
        let plain_schema = r#"{"type":"object","properties":{"name":{"type":"string"}}}"#;
        assert_eq!(source_priority(enum_schema), 0);
        assert_eq!(source_priority(plain_schema), 1);
    }

    #[test]
    fn pair_fingerprint_is_stable_across_object_key_order() {
        let left = canonical_pair_fingerprint(
            r#"{"type":"object","properties":{"name":{"type":"string"}}}"#,
            r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}"#,
            "baseline",
            "jar-hash",
        )
        .unwrap();
        let reordered = canonical_pair_fingerprint(
            r#"{"properties":{"name":{"type":"string"}},"type":"object"}"#,
            r#"{"properties":{"age":{"type":"integer"},"name":{"type":"string"}},"type":"object"}"#,
            "baseline",
            "jar-hash",
        )
        .unwrap();
        assert_eq!(left, reordered);
    }

    #[test]
    fn challenge_reservation_is_independent_from_source_sampling_rank() {
        let roles = (0..128)
            .map(|index| reserve_challenge_family(&format!("family-{index}"), 20260824, 0.20))
            .collect::<Vec<_>>();
        assert!(roles.iter().any(|reserved| *reserved));
        assert!(roles.iter().any(|reserved| !reserved));
    }
}
