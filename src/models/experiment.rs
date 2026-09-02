//! Reproducible, CPU-oriented multi-seed three-way training experiments.
//!
//! This module deliberately records every persisted Softmax model and every
//! scored or traceability-only protocol in one portable report. It does not
//! treat a one-class or structurally unidentifiable challenge as evidence of
//! generalization.

use super::{
    ChallengeProtocol, DatasetRole, DcgPipelineConfig, EvaluationResult,
    FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_SHA256, GeneratedRecordProvenance, ModelArtifact,
    ModelConfig, PreparedDcgDataset, PreparedDcgRecord, TargetMode, ThreeWayEvaluation,
    ThreeWayModelArtifact, TrainingInputProvenance, TrainingMetadata, evaluate_three_way,
    run_three_way_compatibility_pipeline,
};
use crate::dataset::DatasetSplitConfig;
use crate::evaluation::{ClassificationMetrics, evaluate_binary_classification};
use crate::features::{
    DCG_FEATURE_V4_VERSION, DCG_FEATURE_V5_VERSION, DCG_FEATURE_V6_VERSION, POLICY_ACTION_COUNT,
    POLICY_RULE_ACTION_FEATURE_COUNT, POLICY_RULE_IDS,
};
use crate::nn::Sgd;
use crate::prediction::PredictionKind;
use crate::training::TrainingConfig;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Stable JSON format produced by a multi-seed three-way experiment.
pub const THREE_WAY_EXPERIMENT_FORMAT_VERSION: &str = "dcg-three-way-experiment-v1";
/// Stable JSON format for the direction-specific two-label experiment.
pub const BINARY_THREE_SEED_EXPERIMENT_FORMAT_VERSION: &str = "dcg-binary-three-seed-experiment-v1";

/// Stable representation of the feature-space audit that accompanies unusually
/// clean held-out challenge scores. The prepared artifact retains features and
/// pair fingerprints, not raw source schemas, so this audit intentionally
/// reports observable-feature overlap rather than claiming schema-text
/// similarity.
pub const CHALLENGE_NEAR_DUPLICATE_AUDIT_FORMAT_VERSION: &str =
    "dcg-challenge-near-duplicate-audit-v1";

/// A portable audit across selected held-out challenge protocols.
#[derive(Debug, Clone, Serialize)]
pub struct ChallengeNearDuplicateAuditReport {
    pub format_version: String,
    pub dataset_version: String,
    pub feature_version: String,
    pub limitation: String,
    pub protocols: Vec<ChallengeProtocolNearDuplicateAudit>,
}

/// Evidence of exact and near observable-feature overlap for one challenge
/// protocol and its corresponding allowed standard-role training records.
#[derive(Debug, Clone, Serialize)]
pub struct ChallengeProtocolNearDuplicateAudit {
    pub protocol: String,
    pub training_records: usize,
    pub challenge_records: usize,
    pub family_leakage: bool,
    pub pair_fingerprint_leakage: bool,
    pub exact_model_feature_overlap: FeatureOverlapSummary,
    pub exact_policy_free_structural_overlap: FeatureOverlapSummary,
    /// For every challenge record, the number of differing coordinates in its
    /// nearest label-free structural V6 feature vector in allowed training.
    pub nearest_structural_coordinate_distance: BTreeMap<usize, usize>,
    /// Records with a nearest label-free structural feature vector differing in
    /// no more than one coordinate. This threshold is descriptive only; it is
    /// not a claim of semantic schema equivalence.
    pub challenge_records_within_one_structural_coordinate: usize,
    /// Deterministic representatives, prioritizing exact full-model-input
    /// overlap and then closest label-free structural overlap.
    pub samples: Vec<NearDuplicateSample>,
}

/// Counts challenge records that have at least one allowed training record
/// with the same selected feature signature.
#[derive(Debug, Clone, Serialize)]
pub struct FeatureOverlapSummary {
    pub challenge_records_with_match: usize,
    pub challenge_families_with_match: usize,
    pub distinct_training_feature_signatures: usize,
}

/// One deterministic challenge/training comparison selected by the audit.
#[derive(Debug, Clone, Serialize)]
pub struct NearDuplicateSample {
    pub challenge_record_id: String,
    pub challenge_family_id: String,
    pub training_record_id: String,
    pub training_family_id: String,
    pub exact_model_feature_match: bool,
    pub exact_policy_free_structural_match: bool,
    pub structural_coordinate_distance: usize,
}

/// Identifies the independently held-out structural form represented by a
/// generated record. Optional additions are grouped by their label-free root
/// object profile rather than field type, because `open`/`closed` is the
/// actual V6 structural distinction under evaluation.
pub fn structural_variant_key(generation: &GeneratedRecordProvenance) -> Option<String> {
    if generation.declared_mutation == "optional_field_added" {
        return generation
            .root_object_profile
            .as_ref()
            .map(|profile| format!("root-{profile}"));
    }
    (!generation.mutation_variant.trim().is_empty()).then(|| generation.mutation_variant.clone())
}

/// Audits selected held-out challenge protocols for exact identity leakage and
/// feature-space overlap with the records that the corresponding experiment
/// model is allowed to train on. It deliberately does not read labels,
/// predictions, or oracle results.
pub fn audit_challenge_near_duplicates(
    dataset: &PreparedDcgDataset,
    protocols: &[String],
    max_samples_per_protocol: usize,
) -> Result<ChallengeNearDuplicateAuditReport, String> {
    if dataset.feature_version() != DCG_FEATURE_V6_VERSION {
        return Err(format!(
            "near-duplicate audit currently requires {DCG_FEATURE_V6_VERSION}, got {}",
            dataset.feature_version()
        ));
    }
    if protocols.is_empty() {
        return Err("near-duplicate audit requires at least one --protocol".to_owned());
    }
    if max_samples_per_protocol == 0 {
        return Err("near-duplicate audit --max-samples must be positive".to_owned());
    }
    let mut requested = protocols.to_vec();
    requested.sort();
    requested.dedup();
    let protocols = requested
        .iter()
        .map(|protocol| audit_one_challenge_protocol(dataset, protocol, max_samples_per_protocol))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ChallengeNearDuplicateAuditReport {
        format_version: CHALLENGE_NEAR_DUPLICATE_AUDIT_FORMAT_VERSION.to_owned(),
        dataset_version: dataset.dataset_version().to_owned(),
        feature_version: dataset.feature_version().to_owned(),
        limitation: "The portable prepared dataset does not retain raw source schemas. Exact pair fingerprints and family IDs detect identity leakage; feature overlap and coordinate distance describe only the model-visible V6 representation, not semantic equivalence of schema text.".to_owned(),
        protocols,
    })
}

fn audit_one_challenge_protocol(
    dataset: &PreparedDcgDataset,
    protocol: &str,
    max_samples: usize,
) -> Result<ChallengeProtocolNearDuplicateAudit, String> {
    let selection = ChallengeAuditSelection::parse(protocol)?;
    let training = dataset
        .records()
        .iter()
        .filter(|record| {
            record.dataset_role == DatasetRole::Standard && selection.keep_training(record)
        })
        .collect::<Vec<_>>();
    let challenge = dataset
        .records()
        .iter()
        .filter(|record| {
            record.dataset_role == DatasetRole::Challenge && selection.selects_challenge(record)
        })
        .collect::<Vec<_>>();
    if training.is_empty() {
        return Err(format!(
            "{protocol} leaves no standard-role training records"
        ));
    }
    if challenge.is_empty() {
        return Err(format!("{protocol} has no reserved challenge records"));
    }

    let training_families = training
        .iter()
        .map(|record| record.family_id.as_str())
        .collect::<BTreeSet<_>>();
    let challenge_families = challenge
        .iter()
        .map(|record| record.family_id.as_str())
        .collect::<BTreeSet<_>>();
    let training_pairs = training
        .iter()
        .filter_map(|record| record.generation.as_ref())
        .map(|generation| generation.pair_fingerprint.as_str())
        .filter(|fingerprint| !fingerprint.is_empty())
        .collect::<BTreeSet<_>>();
    let challenge_pairs = challenge
        .iter()
        .filter_map(|record| record.generation.as_ref())
        .map(|generation| generation.pair_fingerprint.as_str())
        .filter(|fingerprint| !fingerprint.is_empty())
        .collect::<BTreeSet<_>>();

    let mut full_training = BTreeMap::<Vec<u64>, Vec<&PreparedDcgRecord>>::new();
    let mut structural_training = BTreeMap::<Vec<u64>, Vec<&PreparedDcgRecord>>::new();
    for record in &training {
        full_training
            .entry(full_feature_signature(record))
            .or_default()
            .push(*record);
        structural_training
            .entry(policy_free_structural_signature(record)?)
            .or_default()
            .push(*record);
    }
    let structural_signatures = structural_training.keys().cloned().collect::<Vec<_>>();

    let mut full_match_families = BTreeSet::new();
    let mut structural_match_families = BTreeSet::new();
    let mut full_match_records = 0usize;
    let mut structural_match_records = 0usize;
    let mut within_one = 0usize;
    let mut nearest_distances = BTreeMap::<usize, usize>::new();
    let mut samples = Vec::with_capacity(challenge.len());
    for record in challenge {
        let full_signature = full_feature_signature(record);
        let structural_signature = policy_free_structural_signature(record)?;
        let full_matches = full_training.get(&full_signature);
        let structural_matches = structural_training.get(&structural_signature);
        let exact_full = full_matches.is_some();
        let exact_structural = structural_matches.is_some();
        if exact_full {
            full_match_records += 1;
            full_match_families.insert(record.family_id.clone());
        }
        if exact_structural {
            structural_match_records += 1;
            structural_match_families.insert(record.family_id.clone());
        }
        let (distance, nearest_signature) = structural_signatures
            .iter()
            .map(|candidate| {
                (
                    structural_coordinate_distance(&structural_signature, candidate),
                    candidate,
                )
            })
            .min_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(right.1)))
            .ok_or_else(|| format!("{protocol} has no structural training signatures"))?;
        *nearest_distances.entry(distance).or_default() += 1;
        if distance <= 1 {
            within_one += 1;
        }
        let nearest_record = full_matches
            .and_then(|records| records.first())
            .copied()
            .or_else(|| {
                structural_matches
                    .and_then(|records| records.first())
                    .copied()
            })
            .or_else(|| {
                structural_training
                    .get(nearest_signature)
                    .and_then(|records| records.first())
                    .copied()
            })
            .ok_or_else(|| format!("{protocol} could not select a nearest training record"))?;
        samples.push((
            if exact_full {
                0usize
            } else if exact_structural {
                1usize
            } else {
                2usize
            },
            distance,
            NearDuplicateSample {
                challenge_record_id: record.record_id.clone(),
                challenge_family_id: record.family_id.clone(),
                training_record_id: nearest_record.record_id.clone(),
                training_family_id: nearest_record.family_id.clone(),
                exact_model_feature_match: exact_full,
                exact_policy_free_structural_match: exact_structural,
                structural_coordinate_distance: distance,
            },
        ));
    }
    samples.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.challenge_record_id.cmp(&right.2.challenge_record_id))
            .then_with(|| left.2.training_record_id.cmp(&right.2.training_record_id))
    });
    Ok(ChallengeProtocolNearDuplicateAudit {
        protocol: protocol.to_owned(),
        training_records: training.len(),
        challenge_records: samples.len(),
        family_leakage: !training_families.is_disjoint(&challenge_families),
        pair_fingerprint_leakage: !training_pairs.is_disjoint(&challenge_pairs),
        exact_model_feature_overlap: FeatureOverlapSummary {
            challenge_records_with_match: full_match_records,
            challenge_families_with_match: full_match_families.len(),
            distinct_training_feature_signatures: full_training.len(),
        },
        exact_policy_free_structural_overlap: FeatureOverlapSummary {
            challenge_records_with_match: structural_match_records,
            challenge_families_with_match: structural_match_families.len(),
            distinct_training_feature_signatures: structural_training.len(),
        },
        nearest_structural_coordinate_distance: nearest_distances,
        challenge_records_within_one_structural_coordinate: within_one,
        samples: samples
            .into_iter()
            .take(max_samples)
            .map(|(_, _, sample)| sample)
            .collect(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ChallengeAuditSelection {
    Policy(String),
    Mutation(String),
    MutationVariant { mutation: String, variant: String },
}

impl ChallengeAuditSelection {
    fn parse(protocol: &str) -> Result<Self, String> {
        if let Some(policy) = protocol.strip_prefix("held-out-policy:") {
            return (!policy.is_empty())
                .then(|| Self::Policy(policy.to_owned()))
                .ok_or_else(|| "held-out-policy protocol requires a policy name".to_owned());
        }
        if let Some(rest) = protocol.strip_prefix("held-out-mutation-variant:") {
            let (mutation, variant) = rest.split_once(':').ok_or_else(|| {
                "held-out-mutation-variant protocol must be mutation:variant".to_owned()
            })?;
            return (!mutation.is_empty() && !variant.is_empty())
                .then(|| Self::MutationVariant {
                    mutation: mutation.to_owned(),
                    variant: variant.to_owned(),
                })
                .ok_or_else(|| {
                    "held-out-mutation-variant protocol requires both mutation and variant"
                        .to_owned()
                });
        }
        if let Some(mutation) = protocol.strip_prefix("held-out-mutation:") {
            return (!mutation.is_empty())
                .then(|| Self::Mutation(mutation.to_owned()))
                .ok_or_else(|| "held-out-mutation protocol requires a mutation name".to_owned());
        }
        Err(format!(
            "unsupported audit protocol `{protocol}`; expected held-out-policy:<name>, held-out-mutation:<name>, or held-out-mutation-variant:<mutation>:<variant>"
        ))
    }

    fn keep_training(&self, record: &PreparedDcgRecord) -> bool {
        match self {
            Self::Policy(policy) => record.policy_pack != *policy,
            Self::Mutation(mutation) => record
                .generation
                .as_ref()
                .is_none_or(|generation| generation.declared_mutation != *mutation),
            Self::MutationVariant { mutation, variant } => {
                record.generation.as_ref().is_none_or(|generation| {
                    generation.declared_mutation != *mutation
                        || structural_variant_key(generation).as_deref() != Some(variant.as_str())
                })
            }
        }
    }

    fn selects_challenge(&self, record: &PreparedDcgRecord) -> bool {
        match self {
            Self::Policy(policy) => record.policy_pack == *policy,
            Self::Mutation(mutation) => record
                .generation
                .as_ref()
                .is_some_and(|generation| generation.declared_mutation == *mutation),
            Self::MutationVariant { mutation, variant } => {
                record.generation.as_ref().is_some_and(|generation| {
                    generation.declared_mutation == *mutation
                        && structural_variant_key(generation).as_deref() == Some(variant.as_str())
                })
            }
        }
    }
}

fn full_feature_signature(record: &PreparedDcgRecord) -> Vec<u64> {
    record
        .features
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn policy_free_structural_signature(record: &PreparedDcgRecord) -> Result<Vec<u64>, String> {
    if record.features.len() != 76 {
        return Err(format!(
            "record {} has {} features; expected 76 V6 features",
            record.record_id,
            record.features.len()
        ));
    }
    Ok(record
        .features
        .iter()
        .enumerate()
        .filter(|(index, _)| !(24..48).contains(index) && !(59..62).contains(index))
        .map(|(_, value)| value.to_bits())
        .collect())
}

fn structural_coordinate_distance(left: &[u64], right: &[u64]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(left, right)| left != right)
        .count()
}

/// Validated configuration for the reproducible CPU experiment runner.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreeWayExperimentConfig {
    /// Family-aware split seeds to train and report independently.
    pub seeds: Vec<u64>,
    /// Number of CPU training epochs for every fitted protocol model.
    pub epochs: usize,
    /// Mini-batch size for every fitted protocol model.
    pub batch_size: usize,
    /// SGD learning rate for every fitted protocol model.
    pub learning_rate: f64,
}

impl ThreeWayExperimentConfig {
    /// Validates and canonicalizes the supplied experiment parameters.
    pub fn new(
        mut seeds: Vec<u64>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
    ) -> Result<Self, String> {
        seeds.sort_unstable();
        seeds.dedup();
        if seeds.is_empty() {
            return Err("three-way experiment requires at least one split seed".to_owned());
        }
        TrainingConfig::new(epochs, batch_size).map_err(|error| error.to_string())?;
        Sgd::new(learning_rate).map_err(|error| error.to_string())?;
        Ok(Self {
            seeds,
            epochs,
            batch_size,
            learning_rate,
        })
    }
}

/// Complete report saved beside the per-protocol model artifacts.
#[derive(Debug, Clone, Serialize)]
pub struct ThreeWayExperimentReport {
    /// Format identifier for this report.
    pub format_version: String,
    /// Version stored in the prepared dataset.
    pub dataset_version: String,
    /// Feature schema required by every artifact.
    pub feature_version: String,
    /// Exact dataset, oracle JAR, and policy-pack identities.
    pub input_provenance: TrainingInputProvenance,
    /// Replayed hyperparameters and sorted split seeds.
    pub configuration: ExperimentConfigurationReport,
    /// Independent model fits and protocol results for each seed.
    pub seed_runs: Vec<ThreeWaySeedRunReport>,
    /// Accuracy and confusion totals aggregated only over scored protocols.
    pub aggregate_protocols: Vec<ThreeWayProtocolAggregate>,
}

/// JSON-safe snapshot of the experiment configuration.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentConfigurationReport {
    pub seeds: Vec<u64>,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub split_ratios: [f64; 3],
    pub architecture: Vec<usize>,
    pub objective: String,
    pub execution_device: String,
}

/// Results produced for one deterministic split seed.
#[derive(Debug, Clone, Serialize)]
pub struct ThreeWaySeedRunReport {
    pub seed: u64,
    pub protocols: Vec<ThreeWayProtocolResult>,
}

/// One scored or traceability-only evaluation protocol.
#[derive(Debug, Clone, Serialize)]
pub struct ThreeWayProtocolResult {
    /// Stable protocol name, including an explicit held-out policy/mutation.
    pub protocol: String,
    /// `scored`, `traceability_only`, or `structurally_unidentifiable`.
    pub status: String,
    /// Explanation for a non-scored protocol; absent for a valid score.
    pub reason: Option<String>,
    /// Count of standard-role records allowed to train this protocol model.
    pub training_records: usize,
    /// Family-split validation records used for per-epoch loss only.
    pub validation_records: Option<usize>,
    /// Isolated test or challenge records represented by `metrics`.
    pub evaluation_records: usize,
    /// Canonical label distribution in the isolated evaluation data.
    pub evaluation_labels: BTreeMap<String, usize>,
    /// Final accuracy and confusion matrix when the protocol is scored.
    pub metrics: Option<ThreeWayMetrics>,
    /// Relative path of the fitted portable model artifact, when one exists.
    pub model_artifact: Option<String>,
    /// Per-epoch training loss for the model associated with this protocol.
    pub training_loss: Vec<f64>,
    /// Per-epoch validation loss for the model associated with this protocol.
    pub validation_loss: Vec<f64>,
}

/// Serializable three-way evaluation results.
#[derive(Debug, Clone, Serialize)]
pub struct ThreeWayMetrics {
    /// Fraction correct.
    pub accuracy: f64,
    /// Rows are actual SAFE/WARNING/BREAKING; columns are predicted in that order.
    pub confusion_matrix: [[usize; 3]; 3],
}

/// Aggregate values across split seeds for one identically named protocol.
#[derive(Debug, Clone, Serialize)]
pub struct ThreeWayProtocolAggregate {
    pub protocol: String,
    pub scored_runs: usize,
    pub mean_accuracy: f64,
    pub minimum_accuracy: f64,
    pub maximum_accuracy: f64,
    pub summed_confusion_matrix: [[usize; 3]; 3],
}

/// Portable metrics for the SAFE-versus-BREAKING track. WARNING is absent
/// under the pinned optional-addition policies and is never fabricated.
#[derive(Debug, Clone, Serialize)]
pub struct BinaryExperimentMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    /// Rows are actual non-breaking/BREAKING; columns are predicted in that order.
    pub confusion_matrix: [[usize; 2]; 2],
}

impl From<ClassificationMetrics> for BinaryExperimentMetrics {
    fn from(metrics: ClassificationMetrics) -> Self {
        Self {
            accuracy: metrics.accuracy,
            precision: metrics.precision,
            recall: metrics.recall,
            f1_score: metrics.f1_score,
            confusion_matrix: [
                [
                    metrics.confusion_matrix.true_negative(),
                    metrics.confusion_matrix.false_positive(),
                ],
                [
                    metrics.confusion_matrix.false_negative(),
                    metrics.confusion_matrix.true_positive(),
                ],
            ],
        }
    }
}

/// One independent family split and model fit.
#[derive(Debug, Clone, Serialize)]
pub struct BinarySeedRunReport {
    pub seed: u64,
    pub training_records: usize,
    pub validation_records: usize,
    pub test_records: usize,
    pub challenge_records: usize,
    pub test_metrics: BinaryExperimentMetrics,
    pub challenge_metrics: BinaryExperimentMetrics,
    pub training_loss: Vec<f64>,
    pub validation_loss: Vec<f64>,
    pub model_artifact: String,
    pub model_sha256: String,
}

/// Complete three-seed result for the two classes the pinned oracle actually
/// supplies in this direction-specific corpus.
#[derive(Debug, Clone, Serialize)]
pub struct BinaryThreeSeedExperimentReport {
    pub format_version: String,
    pub dataset_version: String,
    pub feature_version: String,
    pub target: String,
    pub class_order: [String; 2],
    pub input_provenance: TrainingInputProvenance,
    pub configuration: ExperimentConfigurationReport,
    pub seed_runs: Vec<BinarySeedRunReport>,
    pub mean_test_accuracy: f64,
    pub mean_challenge_accuracy: f64,
}

/// Runs the family-isolated binary experiment only after the persisted
/// `benchmark_ready` decision is present and true. Gate checks precede output
/// directory creation and therefore precede training or artifact writes.
pub fn run_binary_three_seed_experiment(
    dataset: &PreparedDcgDataset,
    input_provenance: TrainingInputProvenance,
    configuration: ThreeWayExperimentConfig,
    output_dir: &Path,
) -> Result<BinaryThreeSeedExperimentReport, String> {
    if input_provenance.policy_packs_sha256 == FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_SHA256 {
        return Err(
            "binary three-seed experiment refused: the FIELD_REMOVED severity audit fixture is mechanism-only"
                .to_owned(),
        );
    }
    dataset
        .require_benchmark_ready()
        .map_err(|reason| format!("binary three-seed experiment refused: {reason}"))?;
    let standard = standard_records(dataset, |_| true)?;
    let challenge = PreparedDcgDataset::new(
        format!("{}-challenge", dataset.dataset_version()),
        dataset
            .records()
            .iter()
            .filter(|record| record.dataset_role == DatasetRole::Challenge)
            .cloned()
            .collect(),
    )
    .map_err(|error| error.to_string())?;
    let readiness = standard.training_readiness(
        TargetMode::BinaryBreaking,
        DatasetSplitConfig::new(0.70, 0.15, 0.15, configuration.seeds[0])
            .map_err(|error| error.to_string())?,
    );
    if !readiness.ready_for_training {
        return Err(format!(
            "binary standard training data is not ready: {:?}",
            readiness.reasons
        ));
    }
    let model_directory = output_dir.join("models");
    fs::create_dir_all(&model_directory).map_err(|error| error.to_string())?;
    let mut seed_runs = Vec::with_capacity(configuration.seeds.len());
    for &seed in &configuration.seeds {
        let split =
            DatasetSplitConfig::new(0.70, 0.15, 0.15, seed).map_err(|error| error.to_string())?;
        let pipeline = DcgPipelineConfig::new(
            PredictionKind::BreakingChange,
            split,
            ModelConfig::new(standard.records()[0].features.len(), vec![12, 6], 0.5)
                .map_err(|error| error.to_string())?,
            TrainingConfig::new(configuration.epochs, configuration.batch_size)
                .map_err(|error| error.to_string())?,
            vec![0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        )
        .map_err(|error| error.to_string())?;
        let result = pipeline
            .run(
                &standard,
                Sgd::new(configuration.learning_rate).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?;
        let test_metrics = match result.evaluation {
            EvaluationResult::Classification(metrics) => metrics,
            EvaluationResult::Regression(_) => {
                return Err("binary experiment unexpectedly produced regression metrics".to_owned());
            }
        };
        let challenge_raw = challenge
            .to_dataset(PredictionKind::BreakingChange)
            .map_err(|error| error.to_string())?;
        let challenge_normalized = result
            .scaler
            .transform_dataset(&challenge_raw)
            .map_err(|error| error.to_string())?;
        let challenge_predictions = result
            .model
            .predict_batch(challenge_normalized.features())
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|prediction| {
                prediction
                    .label()
                    .map(f64::from)
                    .ok_or("binary challenge produced a non-classification prediction".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let challenge_targets = challenge_normalized
            .targets()
            .iter()
            .map(|target| {
                (target.len() == 1)
                    .then_some(target[0])
                    .ok_or_else(|| "binary challenge target must contain one value".to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let challenge_metrics =
            evaluate_binary_classification(&challenge_predictions, &challenge_targets)
                .map_err(|error| error.to_string())?;
        let training_records = result.split.train().len();
        let validation_records = result.split.validation().len();
        let test_records = result.split.test().len();
        let training_loss = result.training_history.epoch_losses().to_vec();
        let validation_loss = result.training_history.validation_losses().to_vec();
        let model_name = format!("seed-{seed}-binary-breaking.json");
        let model_path = model_directory.join(&model_name);
        let metadata = TrainingMetadata {
            optimizer: "sgd".to_owned(),
            epochs: configuration.epochs,
            batch_size: configuration.batch_size,
            learning_rate: Some(configuration.learning_rate.to_string()),
            seed,
            training_samples: training_records,
            validation_samples: validation_records,
            test_samples: test_records,
            dataset_id: dataset.dataset_version().to_owned(),
        }
        .validate()
        .map_err(|error| error.to_string())?;
        ModelArtifact::new_with_feature_version(
            "dcg-forward-full-optional-binary-v1",
            dataset.feature_version(),
            result.model,
            result.scaler,
            metadata,
        )
        .map_err(|error| error.to_string())?
        .save(&model_path)
        .map_err(|error| error.to_string())?;
        let model_bytes = fs::read(&model_path).map_err(|error| error.to_string())?;
        seed_runs.push(BinarySeedRunReport {
            seed,
            training_records,
            validation_records,
            test_records,
            challenge_records: challenge.len(),
            test_metrics: test_metrics.into(),
            challenge_metrics: challenge_metrics.into(),
            training_loss,
            validation_loss,
            model_artifact: format!("models/{model_name}"),
            model_sha256: format!("{:x}", Sha256::digest(&model_bytes)),
        });
    }
    let mean_test_accuracy = seed_runs
        .iter()
        .map(|run| run.test_metrics.accuracy)
        .sum::<f64>()
        / seed_runs.len() as f64;
    let mean_challenge_accuracy = seed_runs
        .iter()
        .map(|run| run.challenge_metrics.accuracy)
        .sum::<f64>()
        / seed_runs.len() as f64;
    Ok(BinaryThreeSeedExperimentReport {
        format_version: BINARY_THREE_SEED_EXPERIMENT_FORMAT_VERSION.to_owned(),
        dataset_version: dataset.dataset_version().to_owned(),
        feature_version: dataset.feature_version().to_owned(),
        target: "binary_breaking".to_owned(),
        class_order: ["non_breaking".to_owned(), "breaking".to_owned()],
        input_provenance,
        configuration: ExperimentConfigurationReport {
            seeds: configuration.seeds,
            epochs: configuration.epochs,
            batch_size: configuration.batch_size,
            learning_rate: configuration.learning_rate,
            split_ratios: [0.70, 0.15, 0.15],
            architecture: vec![standard.records()[0].features.len(), 12, 6, 1],
            objective: "binary_cross_entropy".to_owned(),
            execution_device: "cpu".to_owned(),
        },
        seed_runs,
        mean_test_accuracy,
        mean_challenge_accuracy,
    })
}

/// Runs and persists every normal and challenge evaluation across the supplied
/// deterministic seeds. `output_dir` must be a fresh directory owned by the
/// caller; the function creates a `models/` child beneath it.
pub fn run_three_way_experiment(
    dataset: &PreparedDcgDataset,
    input_provenance: TrainingInputProvenance,
    configuration: ThreeWayExperimentConfig,
    output_dir: &Path,
) -> Result<ThreeWayExperimentReport, String> {
    if input_provenance.policy_packs_sha256 == FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_SHA256 {
        return Err(
            "three-way experiment refused: the FIELD_REMOVED severity audit fixture is mechanism-only and cannot be used for corpus or benchmark evidence"
                .to_owned(),
        );
    }
    dataset
        .require_benchmark_ready()
        .map_err(|reason| format!("three-way experiment refused: {reason}"))?;
    let missing_breaking_decisions = dataset
        .records()
        .iter()
        .filter(|record| {
            record.generation.as_ref().is_some_and(|generation| {
                generation.oracle_outcome == "breaking"
                    && generation.oracle_stdout.trim().is_empty()
            })
        })
        .count();
    if missing_breaking_decisions > 0 {
        return Err(format!(
            "three-way experiment is blocked: {missing_breaking_decisions} retained BREAKING records have no oracle decision stdout; regenerate the corpus with the fatal-oracle-failure guard"
        ));
    }
    let standard = standard_records(dataset, |_| true)?;
    if standard.records().is_empty() {
        return Err("three-way experiment requires at least one standard record".to_owned());
    }
    let model_directory = output_dir.join("models");
    fs::create_dir_all(&model_directory).map_err(|error| error.to_string())?;
    let feature_size = standard.records()[0].features.len();
    let readiness = standard.training_readiness(
        TargetMode::ThreeWayCompatibility,
        DatasetSplitConfig::new(0.70, 0.15, 0.15, configuration.seeds[0])
            .map_err(|error| error.to_string())?,
    );
    if !readiness.ready_for_training {
        return Err(format!(
            "three-way standard training data is not ready: {:?}",
            readiness.reasons
        ));
    }
    let mut seed_runs = Vec::with_capacity(configuration.seeds.len());
    for &seed in &configuration.seeds {
        seed_runs.push(run_seed(
            dataset,
            &standard,
            feature_size,
            &input_provenance,
            &configuration,
            &model_directory,
            seed,
        )?);
    }
    let aggregate_protocols = aggregate_protocols(&seed_runs);
    Ok(ThreeWayExperimentReport {
        format_version: THREE_WAY_EXPERIMENT_FORMAT_VERSION.to_owned(),
        dataset_version: dataset.dataset_version().to_owned(),
        feature_version: dataset.feature_version().to_owned(),
        input_provenance,
        configuration: ExperimentConfigurationReport {
            seeds: configuration.seeds,
            epochs: configuration.epochs,
            batch_size: configuration.batch_size,
            learning_rate: configuration.learning_rate,
            split_ratios: [0.70, 0.15, 0.15],
            architecture: vec![feature_size, 12, 6, 3],
            objective: "softmax_cross_entropy".to_owned(),
            execution_device: "cpu".to_owned(),
        },
        seed_runs,
        aggregate_protocols,
    })
}

impl ThreeWayExperimentReport {
    /// Atomically writes the portable experiment report as JSON.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(self).map_err(|error| error.to_string())?;
        let path = path.as_ref();
        let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
        fs::write(&temporary, bytes).map_err(|error| error.to_string())?;
        fs::rename(&temporary, path).map_err(|error| error.to_string())
    }
}

fn run_seed(
    dataset: &PreparedDcgDataset,
    standard: &PreparedDcgDataset,
    feature_size: usize,
    input_provenance: &TrainingInputProvenance,
    configuration: &ThreeWayExperimentConfig,
    model_directory: &Path,
    seed: u64,
) -> Result<ThreeWaySeedRunReport, String> {
    let context = ExperimentRunContext {
        feature_size,
        input_provenance,
        configuration,
        model_directory,
        seed,
    };
    let mut protocols = Vec::new();
    let normal = fit_protocol(
        standard,
        context.feature_size,
        context.input_provenance,
        context.configuration,
        context.model_directory,
        context.seed,
        "normal-family-split",
    )?;
    protocols.push(ThreeWayProtocolResult {
        protocol: "normal-family-split".to_owned(),
        status: "scored".to_owned(),
        reason: None,
        training_records: normal.train_records,
        validation_records: Some(normal.validation_records),
        evaluation_records: normal.test_records,
        evaluation_labels: normal.test_labels.clone(),
        metrics: Some(ThreeWayMetrics::from(normal.test_evaluation)),
        model_artifact: Some(normal.artifact_path.clone()),
        training_loss: normal.training_loss.clone(),
        validation_loss: normal.validation_loss.clone(),
    });

    let policies = standard
        .records()
        .iter()
        .map(|record| record.policy_pack.clone())
        .collect::<BTreeSet<_>>();
    for policy in policies {
        let protocol = format!("held-out-policy:{policy}");
        let challenge = dataset
            .challenge_subset(&ChallengeProtocol::HeldOutPolicy {
                policy_pack: policy.clone(),
            })
            .map_err(|error| error.to_string())?;
        let training = standard_records(dataset, |record| record.policy_pack != policy)?;
        if let Some(reason) = held_out_policy_unidentifiable(&training, &challenge) {
            protocols.push(traceability_result(
                protocol,
                "structurally_unidentifiable",
                &reason,
                training.len(),
                challenge.len(),
                label_counts(&challenge),
            ));
            continue;
        }
        protocols.push(score_challenge_protocol(
            &protocol, training, challenge, &context,
        )?);
    }

    let mutations = standard
        .records()
        .iter()
        .filter_map(|record| {
            record
                .generation
                .as_ref()
                .map(|generation| generation.declared_mutation.clone())
        })
        .collect::<BTreeSet<_>>();
    for mutation in mutations {
        let protocol = format!("held-out-mutation:{mutation}");
        let challenge = dataset
            .challenge_subset(&ChallengeProtocol::HeldOutMutation {
                mutation: mutation.clone(),
            })
            .map_err(|error| error.to_string())?;
        let labels = label_counts(&challenge);
        if matches!(
            mutation.as_str(),
            "enum_value_added" | "optional_field_added"
        ) {
            let description = if mutation == "enum_value_added" {
                "Full ENUM_VALUE_ADDED holdout removes every observed policy-dependent example from training; use its held-out structural-variant protocols for scored interpolation."
            } else {
                "Full OPTIONAL_FIELD_ADDED holdout removes every optional-addition shape from training; it is retained as a zero-support diagnostic, while held-out structural variants measure supported interpolation."
            };
            protocols.push(traceability_result(
                protocol,
                "structurally_unidentifiable",
                description,
                standard.len(),
                challenge.len(),
                labels.clone(),
            ));
            let variants = challenge_variants(&challenge, &mutation);
            if variants.is_empty() {
                protocols.push(traceability_result(
                    format!("held-out-mutation-variant:{mutation}"),
                    "structurally_unidentifiable",
                    "No explicit structural provenance is available for this mutation; regenerate before scoring a structural holdout.",
                    standard.len(),
                    challenge.len(),
                    labels.clone(),
                ));
            }
            for variant in variants {
                let variant_protocol = format!("held-out-mutation-variant:{mutation}:{variant}");
                let variant_challenge = variant_challenge(dataset, &mutation, &variant)?;
                let variant_labels = label_counts(&variant_challenge);
                if !has_peer_variant_in_training(standard, &mutation, &variant) {
                    protocols.push(traceability_result(
                        variant_protocol,
                        "structurally_unidentifiable",
                        "No other structural variant of this mutation is present in training; generate independent variants before treating this as a scored interpolation challenge.",
                        standard.len(),
                        variant_challenge.len(),
                        variant_labels,
                    ));
                    continue;
                }
                if variant_labels.len() < 2 {
                    protocols.push(traceability_result(
                        variant_protocol,
                        "traceability_only",
                        "A one-class challenge cannot measure three-way generalization.",
                        standard.len(),
                        variant_challenge.len(),
                        variant_labels,
                    ));
                    continue;
                }
                let training = standard_records(dataset, |record| {
                    record.generation.as_ref().is_none_or(|generation| {
                        generation.declared_mutation != mutation
                            || structural_variant_key(generation).as_deref()
                                != Some(variant.as_str())
                    })
                })?;
                protocols.push(score_challenge_protocol(
                    &variant_protocol,
                    training,
                    variant_challenge,
                    &context,
                )?);
            }
            continue;
        }
        if labels.len() < 2 {
            protocols.push(traceability_result(
                protocol,
                "traceability_only",
                "A one-class challenge cannot measure three-way generalization.",
                standard.len(),
                challenge.len(),
                labels,
            ));
            continue;
        }
        let training = standard_records(dataset, |record| {
            record
                .generation
                .as_ref()
                .is_none_or(|generation| generation.declared_mutation != mutation)
        })?;
        protocols.push(score_challenge_protocol(
            &protocol, training, challenge, &context,
        )?);
    }

    let paired_challenge = dataset
        .challenge_subset(&ChallengeProtocol::SameMutationDifferentPolicy)
        .map_err(|error| error.to_string())?;
    let paired_labels = label_counts(&paired_challenge);
    if paired_labels.len() < 2 {
        protocols.push(traceability_result(
            "same-mutation-different-policy".to_owned(),
            "traceability_only",
            "A one-class challenge cannot measure three-way generalization.",
            standard.len(),
            paired_challenge.len(),
            paired_labels,
        ));
    } else {
        let metrics = evaluate_challenge(&normal, &paired_challenge)?;
        protocols.push(ThreeWayProtocolResult {
            protocol: "same-mutation-different-policy".to_owned(),
            status: "scored".to_owned(),
            reason: None,
            training_records: normal.train_records,
            validation_records: Some(normal.validation_records),
            evaluation_records: paired_challenge.len(),
            evaluation_labels: paired_labels,
            metrics: Some(metrics),
            model_artifact: Some(normal.artifact_path),
            training_loss: normal.training_loss,
            validation_loss: normal.validation_loss,
        });
    }
    Ok(ThreeWaySeedRunReport { seed, protocols })
}

struct ExperimentRunContext<'a> {
    feature_size: usize,
    input_provenance: &'a TrainingInputProvenance,
    configuration: &'a ThreeWayExperimentConfig,
    model_directory: &'a Path,
    seed: u64,
}

fn score_challenge_protocol(
    protocol: &str,
    training: PreparedDcgDataset,
    challenge: PreparedDcgDataset,
    context: &ExperimentRunContext<'_>,
) -> Result<ThreeWayProtocolResult, String> {
    let fitted = fit_protocol(
        &training,
        context.feature_size,
        context.input_provenance,
        context.configuration,
        context.model_directory,
        context.seed,
        protocol,
    )?;
    let metrics = evaluate_challenge(&fitted, &challenge)?;
    Ok(ThreeWayProtocolResult {
        protocol: protocol.to_owned(),
        status: "scored".to_owned(),
        reason: None,
        training_records: fitted.train_records,
        validation_records: Some(fitted.validation_records),
        evaluation_records: challenge.len(),
        evaluation_labels: label_counts(&challenge),
        metrics: Some(metrics),
        model_artifact: Some(fitted.artifact_path),
        training_loss: fitted.training_loss,
        validation_loss: fitted.validation_loss,
    })
}

fn traceability_result(
    protocol: String,
    status: &str,
    reason: &str,
    training_records: usize,
    evaluation_records: usize,
    evaluation_labels: BTreeMap<String, usize>,
) -> ThreeWayProtocolResult {
    ThreeWayProtocolResult {
        protocol,
        status: status.to_owned(),
        reason: Some(reason.to_owned()),
        training_records,
        validation_records: None,
        evaluation_records,
        evaluation_labels,
        metrics: None,
        model_artifact: None,
        training_loss: Vec::new(),
        validation_loss: Vec::new(),
    }
}

struct FittedProtocol {
    model: super::ThreeWayCompatibilityModel,
    scaler: crate::preprocessing::StandardScaler,
    train_records: usize,
    validation_records: usize,
    test_records: usize,
    test_labels: BTreeMap<String, usize>,
    test_evaluation: ThreeWayEvaluation,
    training_loss: Vec<f64>,
    validation_loss: Vec<f64>,
    artifact_path: String,
}

fn fit_protocol(
    dataset: &PreparedDcgDataset,
    feature_size: usize,
    input_provenance: &TrainingInputProvenance,
    configuration: &ThreeWayExperimentConfig,
    model_directory: &Path,
    seed: u64,
    protocol: &str,
) -> Result<FittedProtocol, String> {
    let result = run_three_way_compatibility_pipeline(
        dataset,
        DatasetSplitConfig::new(0.70, 0.15, 0.15, seed).map_err(|error| error.to_string())?,
        ModelConfig::new(feature_size, vec![12, 6], 0.5).map_err(|error| error.to_string())?,
        TrainingConfig::new(configuration.epochs, configuration.batch_size)
            .map_err(|error| error.to_string())?,
        Sgd::new(configuration.learning_rate).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let artifact_name = format!("seed-{seed}-{}.json", file_stem(protocol));
    let artifact_path = model_directory.join(&artifact_name);
    let feature_generation = dataset
        .feature_version()
        .strip_prefix("dcg-features-")
        .unwrap_or(dataset.feature_version());
    ThreeWayModelArtifact::new_with_feature_version(
        format!(
            "dcg-three-way-{feature_generation}-{}-seed-{seed}",
            file_stem(protocol)
        ),
        dataset.feature_version(),
        result.model.clone(),
        result.scaler.clone(),
        TrainingMetadata {
            optimizer: "sgd".to_owned(),
            epochs: configuration.epochs,
            batch_size: configuration.batch_size,
            learning_rate: Some(configuration.learning_rate.to_string()),
            seed,
            training_samples: result.split.train().len(),
            validation_samples: result.split.validation().len(),
            test_samples: result.split.test().len(),
            dataset_id: dataset.dataset_version().to_owned(),
        },
        input_provenance.clone(),
    )
    .map_err(|error| error.to_string())?
    .save(&artifact_path)
    .map_err(|error| error.to_string())?;
    // Loading immediately makes persistence failures fail the experiment rather
    // than leaving an unverified file in the result directory.
    ThreeWayModelArtifact::load(&artifact_path).map_err(|error| error.to_string())?;
    Ok(FittedProtocol {
        model: result.model,
        scaler: result.scaler,
        train_records: result.split.train().len(),
        validation_records: result.split.validation().len(),
        test_records: result.split.test().len(),
        test_labels: label_counts(result.split.test()),
        test_evaluation: result.evaluation,
        training_loss: result.training_history.epoch_losses().to_vec(),
        validation_loss: result.training_history.validation_losses().to_vec(),
        artifact_path: PathBuf::from("models")
            .join(artifact_name)
            .to_string_lossy()
            .into_owned(),
    })
}

fn evaluate_challenge(
    fitted: &FittedProtocol,
    challenge: &PreparedDcgDataset,
) -> Result<ThreeWayMetrics, String> {
    let raw = challenge
        .to_target_dataset(TargetMode::ThreeWayCompatibility)
        .map_err(|error| error.to_string())?;
    let normalized = fitted
        .scaler
        .transform_dataset(&raw)
        .map_err(|error| error.to_string())?;
    let predictions = normalized
        .features()
        .iter()
        .map(|features| fitted.model.predict(features))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let evaluation = evaluate_three_way(&predictions, normalized.targets())
        .map_err(|error| error.to_string())?;
    Ok(ThreeWayMetrics::from(evaluation))
}

impl From<ThreeWayEvaluation> for ThreeWayMetrics {
    fn from(value: ThreeWayEvaluation) -> Self {
        Self {
            accuracy: value.accuracy,
            confusion_matrix: value.confusion_matrix,
        }
    }
}

fn standard_records(
    dataset: &PreparedDcgDataset,
    keep: impl Fn(&super::PreparedDcgRecord) -> bool,
) -> Result<PreparedDcgDataset, String> {
    PreparedDcgDataset::new(
        format!(
            "{}-three-way-experiment-training",
            dataset.dataset_version()
        ),
        dataset
            .records()
            .iter()
            .filter(|record| record.dataset_role == DatasetRole::Standard && keep(record))
            .cloned()
            .collect(),
    )
    .map_err(|error| error.to_string())
}

fn variant_challenge(
    dataset: &PreparedDcgDataset,
    mutation: &str,
    variant: &str,
) -> Result<PreparedDcgDataset, String> {
    PreparedDcgDataset::new(
        format!(
            "{}-three-way-experiment-challenge-{mutation}-{variant}",
            dataset.dataset_version()
        ),
        dataset
            .records()
            .iter()
            .filter(|record| {
                record.dataset_role == DatasetRole::Challenge
                    && record.generation.as_ref().is_some_and(|generation| {
                        generation.declared_mutation == mutation
                            && structural_variant_key(generation).as_deref() == Some(variant)
                    })
            })
            .cloned()
            .collect(),
    )
    .map_err(|error| error.to_string())
}

fn challenge_variants(challenge: &PreparedDcgDataset, mutation: &str) -> BTreeSet<String> {
    challenge
        .records()
        .iter()
        .filter_map(|record| {
            record.generation.as_ref().and_then(|generation| {
                (generation.declared_mutation == mutation)
                    .then(|| structural_variant_key(generation))
                    .flatten()
            })
        })
        .collect()
}

fn has_peer_variant_in_training(
    training: &PreparedDcgDataset,
    mutation: &str,
    held_out_variant: &str,
) -> bool {
    training.records().iter().any(|record| {
        record.generation.as_ref().is_some_and(|generation| {
            generation.declared_mutation == mutation
                && structural_variant_key(generation).as_deref() != Some(held_out_variant)
        })
    })
}

/// Returns a reason when an all-policy holdout requires an unobserved policy
/// primitive. V4 can only recognize exact enum-action profiles. V5 and V6
/// can also score a deliberately held-out *combination* when every per-rule
/// action in that combination has appeared independently in training; a
/// wholly unseen rule/action remains a zero-shot request and is not scored.
fn held_out_policy_unidentifiable(
    training: &PreparedDcgDataset,
    challenge: &PreparedDcgDataset,
) -> Option<String> {
    if training.feature_version() != challenge.feature_version() {
        return None;
    }
    if matches!(
        training.feature_version(),
        DCG_FEATURE_V5_VERSION | DCG_FEATURE_V6_VERSION
    ) {
        return held_out_v5_policy_action_unidentifiable(training, challenge);
    }
    let training_profiles = training
        .records()
        .iter()
        .filter_map(policy_semantic_profile)
        .collect::<BTreeSet<_>>();
    let held_out_profiles = challenge
        .records()
        .iter()
        .filter_map(policy_semantic_profile)
        .collect::<BTreeSet<_>>();
    if training_profiles.is_empty() || held_out_profiles.is_empty() {
        return None;
    }
    let missing_profiles = held_out_profiles
        .difference(&training_profiles)
        .cloned()
        .collect::<Vec<_>>();
    (!missing_profiles.is_empty()).then(|| {
        format!(
            "Held-out policy has declared policy-semantic profile(s) {missing_profiles:?} absent from training. Add independently named approved packs with the same profile before scoring policy-identity generalization."
        )
    })
}

fn held_out_v5_policy_action_unidentifiable(
    training: &PreparedDcgDataset,
    challenge: &PreparedDcgDataset,
) -> Option<String> {
    let training_actions = training
        .records()
        .iter()
        .filter_map(v5_policy_action_components)
        .flatten()
        .collect::<BTreeSet<_>>();
    let held_out_actions = challenge
        .records()
        .iter()
        .filter_map(v5_policy_action_components)
        .flatten()
        .collect::<BTreeSet<_>>();
    if training_actions.is_empty() || held_out_actions.is_empty() {
        return None;
    }
    let missing_actions = held_out_actions
        .difference(&training_actions)
        .cloned()
        .collect::<Vec<_>>();
    (!missing_actions.is_empty()).then(|| {
        format!(
            "Held-out V5 policy requires declared rule/action component(s) {missing_actions:?} absent from training. This is zero-shot policy behavior, not a supervised compositional evaluation."
        )
    })
}

/// Produces the eight semantic primitives from the V5/V6 24-value resolved
/// policy prefix. Invalid/non-one-hot vectors are ignored here;
/// prepared-dataset validation remains responsible for rejecting malformed
/// persisted records.
fn v5_policy_action_components(record: &super::PreparedDcgRecord) -> Option<Vec<String>> {
    if !matches!(
        record.feature_version.as_str(),
        DCG_FEATURE_V5_VERSION | DCG_FEATURE_V6_VERSION
    ) {
        return None;
    }
    let values = record
        .features
        .iter()
        .skip(24)
        .take(POLICY_RULE_ACTION_FEATURE_COUNT)
        .copied()
        .collect::<Vec<_>>();
    if values.len() != POLICY_RULE_ACTION_FEATURE_COUNT {
        return None;
    }
    POLICY_RULE_IDS
        .iter()
        .enumerate()
        .map(|(rule_index, rule)| {
            let actions =
                &values[rule_index * POLICY_ACTION_COUNT..(rule_index + 1) * POLICY_ACTION_COUNT];
            let action_index = actions.iter().position(|value| *value == 1.0)?;
            (actions.iter().filter(|value| **value == 1.0).count() == 1)
                .then(|| format!("{rule}={}", ["IGNORE", "WARNING", "BREAKING"][action_index]))
        })
        .collect()
}

fn policy_semantic_profile(record: &super::PreparedDcgRecord) -> Option<Vec<u8>> {
    let end = match record.feature_version.as_str() {
        DCG_FEATURE_V4_VERSION => 27,
        DCG_FEATURE_V5_VERSION | DCG_FEATURE_V6_VERSION => 24 + POLICY_RULE_ACTION_FEATURE_COUNT,
        _ => return None,
    };
    Some(
        record
            .features
            .iter()
            .skip(24)
            .take(end - 24)
            .map(|value| u8::from(*value == 1.0))
            .collect(),
    )
}

fn label_counts(dataset: &PreparedDcgDataset) -> BTreeMap<String, usize> {
    let mut labels = BTreeMap::new();
    for record in dataset.records() {
        let label = record
            .compatibility_label
            .map_or_else(|| "missing".to_owned(), |label| label.as_str().to_owned());
        *labels.entry(label).or_default() += 1;
    }
    labels
}

fn file_stem(protocol: &str) -> String {
    protocol
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect()
}

fn aggregate_protocols(seed_runs: &[ThreeWaySeedRunReport]) -> Vec<ThreeWayProtocolAggregate> {
    let mut metrics_by_protocol = BTreeMap::<String, Vec<&ThreeWayMetrics>>::new();
    for seed_run in seed_runs {
        for protocol in &seed_run.protocols {
            if let Some(metrics) = &protocol.metrics {
                metrics_by_protocol
                    .entry(protocol.protocol.clone())
                    .or_default()
                    .push(metrics);
            }
        }
    }
    metrics_by_protocol
        .into_iter()
        .map(|(protocol, metrics)| {
            let accuracies = metrics
                .iter()
                .map(|metrics| metrics.accuracy)
                .collect::<Vec<_>>();
            let mut summed_confusion_matrix = [[0; 3]; 3];
            for metric in metrics {
                for (sum_row, metric_row) in summed_confusion_matrix
                    .iter_mut()
                    .zip(metric.confusion_matrix.iter())
                {
                    for (sum, value) in sum_row.iter_mut().zip(metric_row.iter()) {
                        *sum += *value;
                    }
                }
            }
            ThreeWayProtocolAggregate {
                protocol,
                scored_runs: accuracies.len(),
                mean_accuracy: accuracies.iter().sum::<f64>() / accuracies.len() as f64,
                minimum_accuracy: accuracies.iter().copied().fold(f64::INFINITY, f64::min),
                maximum_accuracy: accuracies.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                summed_confusion_matrix,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::realistic_fixtures;
    use crate::models::{GeneratedRecordProvenance, PreparedDcgRecord};

    #[test]
    fn canonicalizes_seed_order_and_rejects_empty_seed_sets() {
        let config = ThreeWayExperimentConfig::new(vec![9, 2, 9], 1, 1, 0.1).unwrap();
        assert_eq!(config.seeds, vec![2, 9]);
        assert!(ThreeWayExperimentConfig::new(vec![], 1, 1, 0.1).is_err());
    }

    #[test]
    fn runner_rejects_false_or_missing_benchmark_ready_before_training() {
        let configuration = ThreeWayExperimentConfig::new(vec![7], 1, 1, 0.1).unwrap();
        let provenance = TrainingInputProvenance {
            dataset_sha256: "a".repeat(64),
            oracle_jar_sha256: "b".repeat(64),
            policy_packs_sha256: "c".repeat(64),
        };
        let output = std::env::temp_dir().join(format!(
            "dcg-benchmark-ready-runner-test-{}",
            std::process::id()
        ));

        let missing =
            PreparedDcgDataset::from_fixtures("missing-benchmark-ready", &realistic_fixtures())
                .unwrap();
        let audit_fixture_provenance = TrainingInputProvenance {
            dataset_sha256: "a".repeat(64),
            oracle_jar_sha256: "b".repeat(64),
            policy_packs_sha256: FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_SHA256.to_owned(),
        };
        let error = run_three_way_experiment(
            &missing,
            audit_fixture_provenance,
            configuration.clone(),
            &output,
        )
        .unwrap_err();
        assert!(
            error.contains("severity audit fixture is mechanism-only"),
            "{error}"
        );
        let error =
            run_three_way_experiment(&missing, provenance.clone(), configuration.clone(), &output)
                .unwrap_err();
        assert!(error.contains("benchmark_ready is missing"), "{error}");
        let error = run_binary_three_seed_experiment(
            &missing,
            provenance.clone(),
            configuration.clone(),
            &output,
        )
        .unwrap_err();
        assert!(error.contains("benchmark_ready is missing"), "{error}");

        let mut false_ready =
            PreparedDcgDataset::from_fixtures("false-benchmark-ready", &realistic_fixtures())
                .unwrap();
        let readiness = false_ready.stamp_benchmark_readiness(None);
        assert!(!readiness.ready_for_generalization_benchmark);
        let error = run_three_way_experiment(
            &false_ready,
            provenance.clone(),
            configuration.clone(),
            &output,
        )
        .unwrap_err();
        assert!(error.contains("benchmark_ready=false"), "{error}");
        let error =
            run_binary_three_seed_experiment(&false_ready, provenance, configuration, &output)
                .unwrap_err();
        assert!(error.contains("benchmark_ready=false"), "{error}");
        assert!(
            !output.exists(),
            "readiness refusal must precede output creation"
        );
    }

    #[test]
    fn optional_structural_variants_use_root_profile_not_field_type() {
        let mut generation = GeneratedRecordProvenance {
            oracle_jar_sha256: "jar".to_owned(),
            policy_packs_sha256: "packs".to_owned(),
            oracle_outcome: "safe".to_owned(),
            declared_mutation: "optional_field_added".to_owned(),
            mutation_variant: "optional-string-field".to_owned(),
            root_object_profile: Some("open".to_owned()),
            oracle_stdout: "Schema compatibility: PASS\n".to_owned(),
            compatibility_mode: "BACKWARD".to_owned(),
            generator_version: "test".to_owned(),
            generation_seed: 1,
            label_source: "pinned-contract-cli-jar".to_owned(),
            oracle_invariant_rationale: None,
            pair_fingerprint: "pair".to_owned(),
        };
        assert_eq!(
            structural_variant_key(&generation).as_deref(),
            Some("root-open")
        );
        generation.root_object_profile = Some("closed".to_owned());
        assert_eq!(
            structural_variant_key(&generation).as_deref(),
            Some("root-closed")
        );
    }

    #[test]
    fn near_duplicate_audit_reports_feature_overlap_without_identity_leakage() {
        let mut features = vec![0.0; 76];
        features[11] = 1.0;
        features[24 + 4 * 3 + 1] = 1.0;
        let training = v6_audit_record(
            "train-record",
            "train-family",
            DatasetRole::Standard,
            "baseline",
            "pair-train",
            features.clone(),
        );
        let challenge = v6_audit_record(
            "challenge-record",
            "challenge-family",
            DatasetRole::Challenge,
            "held-out-policy",
            "pair-challenge",
            features,
        );
        let dataset = PreparedDcgDataset::new("v6-audit", vec![training, challenge]).unwrap();
        let report = audit_challenge_near_duplicates(
            &dataset,
            &["held-out-policy:held-out-policy".to_owned()],
            3,
        )
        .unwrap();
        let protocol = &report.protocols[0];
        assert!(!protocol.family_leakage);
        assert!(!protocol.pair_fingerprint_leakage);
        assert_eq!(
            protocol
                .exact_model_feature_overlap
                .challenge_records_with_match,
            1
        );
        assert_eq!(
            protocol
                .exact_policy_free_structural_overlap
                .challenge_records_with_match,
            1
        );
        assert_eq!(
            protocol.nearest_structural_coordinate_distance.get(&0),
            Some(&1)
        );
        assert_eq!(protocol.samples[0].training_family_id, "train-family");
    }

    #[test]
    fn aggregates_only_scored_protocols() {
        let reports = vec![
            ThreeWaySeedRunReport {
                seed: 1,
                protocols: vec![ThreeWayProtocolResult {
                    protocol: "normal".to_owned(),
                    status: "scored".to_owned(),
                    reason: None,
                    training_records: 1,
                    validation_records: Some(1),
                    evaluation_records: 1,
                    evaluation_labels: BTreeMap::new(),
                    metrics: Some(ThreeWayMetrics {
                        accuracy: 0.5,
                        confusion_matrix: [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
                    }),
                    model_artifact: None,
                    training_loss: vec![],
                    validation_loss: vec![],
                }],
            },
            ThreeWaySeedRunReport {
                seed: 2,
                protocols: vec![ThreeWayProtocolResult {
                    protocol: "normal".to_owned(),
                    status: "scored".to_owned(),
                    reason: None,
                    training_records: 1,
                    validation_records: Some(1),
                    evaluation_records: 1,
                    evaluation_labels: BTreeMap::new(),
                    metrics: Some(ThreeWayMetrics {
                        accuracy: 1.0,
                        confusion_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    }),
                    model_artifact: None,
                    training_loss: vec![],
                    validation_loss: vec![],
                }],
            },
        ];
        let aggregate = aggregate_protocols(&reports);
        assert_eq!(aggregate.len(), 1);
        assert_eq!(aggregate[0].scored_runs, 2);
        assert_eq!(aggregate[0].mean_accuracy, 0.75);
        assert_eq!(aggregate[0].summed_confusion_matrix[0][0], 2);
    }

    #[test]
    fn identifies_unseen_v4_policy_semantics_without_inspecting_labels() {
        let training =
            PreparedDcgDataset::new("training", vec![v4_record("family-a", "baseline", 24)])
                .unwrap();
        let challenge =
            PreparedDcgDataset::new("challenge", vec![v4_record("family-b", "strict", 26)])
                .unwrap();
        assert!(held_out_policy_unidentifiable(&training, &challenge).is_some());
    }

    #[test]
    fn identifies_unseen_v5_policy_semantics_without_inspecting_labels() {
        let training =
            PreparedDcgDataset::new("training", vec![v5_record("family-a", "baseline", 1, 2)])
                .unwrap();
        let challenge =
            PreparedDcgDataset::new("challenge", vec![v5_record("family-b", "custom", 1, 0)])
                .unwrap();
        assert!(held_out_policy_unidentifiable(&training, &challenge).is_some());
    }

    #[test]
    fn allows_unseen_v5_policy_combinations_when_every_action_was_observed() {
        let training = PreparedDcgDataset::new(
            "training",
            vec![
                v5_record("family-a", "enum-warning-constraint-breaking", 1, 2),
                v5_record("family-b", "enum-breaking-constraint-ignore", 2, 0),
            ],
        )
        .unwrap();
        let challenge = PreparedDcgDataset::new(
            "challenge",
            vec![v5_record(
                "family-c",
                "enum-warning-constraint-ignore",
                1,
                0,
            )],
        )
        .unwrap();
        assert!(held_out_policy_unidentifiable(&training, &challenge).is_none());
    }

    fn v4_record(family: &str, policy_pack: &str, profile_index: usize) -> PreparedDcgRecord {
        let mut record = PreparedDcgRecord::from_fixture(&realistic_fixtures()[0]).unwrap();
        record.record_id = family.to_owned();
        record.family_id = family.to_owned();
        record.split_group_id = family.to_owned();
        record.contract_id = family.to_owned();
        record.policy_pack = policy_pack.to_owned();
        record.feature_version = DCG_FEATURE_V4_VERSION.to_owned();
        let mut features = vec![0.0; 38];
        features[profile_index] = 1.0;
        record.features = crate::linalg::Vector::new(features);
        record
    }

    fn v5_record(
        family: &str,
        policy_pack: &str,
        enum_value_added_action: usize,
        constraint_tightened_action: usize,
    ) -> PreparedDcgRecord {
        let mut record = v4_record(family, policy_pack, 24);
        record.feature_version = DCG_FEATURE_V5_VERSION.to_owned();
        let mut features = vec![0.0; 59];
        for rule_index in 0..8 {
            // The JAR baseline is BREAKING for every rule except enum-value
            // addition, which is WARNING.
            let action = if rule_index == 4 { 1 } else { 2 };
            features[24 + rule_index * 3 + action] = 1.0;
        }
        features[24 + 4 * 3 + 1] = 0.0;
        features[24 + 4 * 3 + enum_value_added_action] = 1.0;
        features[24 + 5 * 3 + 2] = 0.0;
        features[24 + 5 * 3 + constraint_tightened_action] = 1.0;
        record.features = crate::linalg::Vector::new(features);
        record
    }

    fn v6_audit_record(
        record_id: &str,
        family_id: &str,
        dataset_role: DatasetRole,
        policy_pack: &str,
        pair_fingerprint: &str,
        features: Vec<f64>,
    ) -> PreparedDcgRecord {
        let mut record = v4_record(family_id, policy_pack, 24);
        record.record_id = record_id.to_owned();
        record.dataset_role = dataset_role;
        record.feature_version = DCG_FEATURE_V6_VERSION.to_owned();
        record.features = crate::linalg::Vector::new(features);
        record.generation = Some(GeneratedRecordProvenance {
            oracle_jar_sha256: "jar".to_owned(),
            policy_packs_sha256: "packs".to_owned(),
            oracle_outcome: "safe".to_owned(),
            declared_mutation: "enum_value_added".to_owned(),
            mutation_variant: "add-type-preserving-enum-value-0".to_owned(),
            root_object_profile: None,
            oracle_stdout: "Schema compatibility: PASS\n".to_owned(),
            compatibility_mode: "BACKWARD".to_owned(),
            generator_version: "test".to_owned(),
            generation_seed: 1,
            label_source: "pinned-contract-cli-jar".to_owned(),
            oracle_invariant_rationale: None,
            pair_fingerprint: pair_fingerprint.to_owned(),
        });
        record
    }
}
