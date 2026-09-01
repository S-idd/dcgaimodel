//! Frozen-model evaluation of independently sourced DCG schema transitions.
//!
//! This module deliberately separates external evidence from training corpus
//! construction. It labels every manifest transition with the pinned oracle,
//! rejects V9 identity or near-feature overlap before inference, and uses an
//! already-fitted V9 scaler/model without training, refitting, or persistence
//! changes to either input artifact.

use super::{CompatibilityLabel, PreparedDcgDataset, ThreeWayModelArtifact};
use crate::features::{
    ApprovedPolicyContexts, DCG_FEATURE_V6_NAMES, DCG_FEATURE_V6_VERSION,
    SchemaChangeFeatureV6Extractor,
};
use crate::generation::{
    GeneratedPair, OracleConfig, OracleOutcome, PinnedOracle, canonical_pair_fingerprint,
};
use crate::linalg::Vector;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

/// Required format identifier for external transition manifests.
pub const EXTERNAL_TRANSITION_MANIFEST_FORMAT_VERSION: &str = "dcg-external-transition-manifest-v3";
/// Stable report format for frozen external evaluation evidence.
pub const EXTERNAL_EVALUATION_REPORT_FORMAT_VERSION: &str = "dcg-external-evaluation-report-v3";

/// Declared origin class for independently sourced evaluation transitions.
/// Internal seed or conformance fixtures are intentionally not an option.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ExternalSourceKind {
    PublicVersionHistory,
    PublicSchemaRepository,
    ManualUnrelated,
}

/// Declares whether an external transition may contribute to an accuracy
/// metric. This is intentionally part of the validated manifest schema, not
/// an annotation that callers may ignore.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ExternalUse {
    AccuracyBearing,
    TraceabilityOnly,
    RejectedV9Overlap,
}

impl ExternalUse {
    fn is_accuracy_bearing(self) -> bool {
        self == Self::AccuracyBearing
    }
}

/// Inputs that bind an external evaluation to one frozen V9 model and corpus.
#[derive(Debug, Clone)]
pub struct ExternalEvaluationConfig {
    pub manifest_path: PathBuf,
    pub v9_dataset_path: PathBuf,
    pub model_path: PathBuf,
    pub oracle_config: OracleConfig,
    pub oracle_workspace: PathBuf,
    /// Reject a record when its nearest policy-free structural V6 vector in V9
    /// differs in at most this many coordinates. The recommended default is 1.
    pub max_structural_coordinate_distance: usize,
}

/// No-oracle input set for manifest validation and V9 contamination checks.
#[derive(Debug, Clone)]
pub struct ExternalManifestPreflightConfig {
    pub manifest_path: PathBuf,
    pub v9_dataset_path: PathBuf,
    pub jar_path: PathBuf,
    pub policy_packs_path: PathBuf,
    pub max_structural_coordinate_distance: usize,
}

/// Read-only inputs for inspecting one external transition's V6 feature-space
/// relationship to V9. This never launches Java, hashes a JAR, or loads a
/// fitted model/scaler.
#[derive(Debug, Clone)]
pub struct ExternalFeatureDiagnosticConfig {
    pub manifest_path: PathBuf,
    pub v9_dataset_path: PathBuf,
    pub policy_packs_path: PathBuf,
    pub record_id: String,
    pub nearest_record_limit: usize,
}

/// One original V6 coordinate, retained with its stable feature name.
#[derive(Debug, Clone, Serialize)]
pub struct ExternalFeatureCoordinate {
    pub index: usize,
    pub name: String,
    pub value: f64,
}

/// One policy-free coordinate that differs between the inspected transition
/// and a nearest V9 record.
#[derive(Debug, Clone, Serialize)]
pub struct ExternalFeatureCoordinateDifference {
    pub index: usize,
    pub name: String,
    pub external_value: f64,
    pub v9_value: f64,
}

/// A V9 record tied for the minimum policy-free structural distance.
#[derive(Debug, Clone, Serialize)]
pub struct ExternalFeatureDiagnosticNeighbor {
    pub record_id: String,
    pub source: String,
    pub family_id: String,
    pub policy_pack: String,
    pub declared_mutation: Option<String>,
    pub mutation_variant: Option<String>,
    pub pair_fingerprint: Option<String>,
    pub structural_coordinate_distance: usize,
    pub differing_structural_coordinates: Vec<ExternalFeatureCoordinateDifference>,
}

/// Read-only, auditable explanation of one external record's V6 feature
/// extraction and its closest V9 feature-space neighbour(s).
#[derive(Debug, Clone, Serialize)]
pub struct ExternalFeatureDiagnosticReport {
    pub format_version: String,
    pub oracle_invoked: bool,
    pub model_inference_invoked: bool,
    pub labels_produced: bool,
    pub record_id: String,
    pub source: String,
    pub source_path: String,
    pub mutation_id: String,
    pub policy_pack: String,
    pub full_v6_feature_count: usize,
    pub structural_coordinate_count: usize,
    pub structural_coordinates: Vec<ExternalFeatureCoordinate>,
    pub nonzero_v6_features: Vec<ExternalFeatureCoordinate>,
    pub exact_full_v6_feature_match_count: usize,
    pub nearest_v9_structural_distance: usize,
    pub nearest_v9_record_count: usize,
    pub nearest_v9_records: Vec<ExternalFeatureDiagnosticNeighbor>,
}

/// Portable validation result; this operation never launches Java or a model.
#[derive(Debug, Clone, Serialize)]
pub struct ExternalManifestPreflightReport {
    pub format_version: String,
    pub oracle_invoked: bool,
    pub model_inference_invoked: bool,
    pub manifest_sha256: String,
    pub v9_dataset_sha256: String,
    pub oracle_jar_sha256: String,
    pub policy_packs_sha256: String,
    pub max_structural_coordinate_distance: usize,
    pub passed: bool,
    pub total_records: usize,
    pub accepted_records: usize,
    pub rejected_records: usize,
    pub records: Vec<ExternalManifestPreflightRecord>,
}

/// Per-record source/pair/feature-space contamination decision.
#[derive(Debug, Clone, Serialize)]
pub struct ExternalManifestPreflightRecord {
    pub record_id: String,
    pub source: String,
    pub family_id: String,
    pub policy_pack: String,
    pub mutation_id: String,
    pub pair_fingerprint: String,
    pub exact_model_feature_overlap: bool,
    pub nearest_policy_free_structural_distance: usize,
    pub accepted: bool,
    pub rejection_reasons: Vec<String>,
}

/// Portable result of one inference-only external evaluation.
#[derive(Debug, Clone, Serialize)]
pub struct ExternalEvaluationReport {
    pub format_version: String,
    pub mode: String,
    pub frozen_inference_only: bool,
    pub max_structural_coordinate_distance: usize,
    pub v9_reference: ExternalV9Reference,
    pub manifest_sha256: String,
    /// Manifest-native explanation of excluded traceability evidence. This is
    /// copied into the report so consumers need not consult an audit sidecar.
    pub traceability: ExternalTraceabilityMetadata,
    pub total_manifest_records: usize,
    pub scored_records: usize,
    pub traceability_excluded_records: usize,
    pub manifest_overlap_excluded_records: usize,
    pub oracle_rejected_records: usize,
    pub overlap_rejected_records: usize,
    pub overall: ExternalMetricReport,
    pub by_mutation: BTreeMap<String, ExternalMetricReport>,
    pub by_policy_pack: BTreeMap<String, ExternalMetricReport>,
    pub by_mutation_policy_pack: BTreeMap<String, ExternalMetricReport>,
    pub records: Vec<ExternalTransitionEvaluationRecord>,
}

/// Immutable identities that make an external report reproducible and prevent
/// it from silently evaluating a model against mismatched oracle inputs.
#[derive(Debug, Clone, Serialize)]
pub struct ExternalV9Reference {
    pub v9_dataset_path: String,
    pub v9_dataset_sha256: String,
    pub model_path: String,
    pub model_sha256: String,
    pub model_version: String,
    pub feature_version: String,
    pub oracle_jar_sha256: String,
    pub policy_packs_sha256: String,
}

/// One transition's oracle evidence, overlap decision, and optional frozen
/// model prediction. Schema text is intentionally not copied into this report.
#[derive(Debug, Clone, Serialize)]
pub struct ExternalTransitionEvaluationRecord {
    pub record_id: String,
    pub source: String,
    pub source_kind: ExternalSourceKind,
    pub source_url: String,
    pub base_revision: String,
    pub candidate_revision: String,
    pub base_commit: String,
    pub candidate_commit: String,
    pub base_blob_sha1: String,
    pub candidate_blob_sha1: String,
    pub source_license: String,
    pub source_path: String,
    pub family_id: String,
    pub contract_id: String,
    pub old_version: String,
    pub new_version: String,
    pub policy_pack: String,
    pub mutation_id: String,
    pub external_use: ExternalUse,
    pub status: String,
    pub rejection_reasons: Vec<String>,
    pub pair_fingerprint: String,
    pub exact_model_feature_overlap: bool,
    pub nearest_policy_free_structural_distance: Option<usize>,
    pub oracle: ExternalOracleEvidence,
    pub actual_label: Option<String>,
    pub predicted_label: Option<String>,
    pub predicted_probabilities: Option<Vec<f64>>,
}

/// Captured external oracle output. `actual_label` is absent only when the JAR
/// rejects the schema or terminates through its guarded failure path.
#[derive(Debug, Clone, Serialize)]
pub struct ExternalOracleEvidence {
    pub outcome: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

/// Three-class metrics over accepted, non-overlapping external records.
#[derive(Debug, Clone, Serialize)]
pub struct ExternalMetricReport {
    pub records: usize,
    pub accuracy: Option<f64>,
    /// Rows are actual SAFE/WARNING/BREAKING; columns are predicted in that order.
    pub confusion_matrix: [[usize; 3]; 3],
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ExternalTransitionManifest {
    format_version: String,
    summary: ExternalTransitionManifestSummary,
    transitions: Vec<ExternalTransitionInput>,
}

/// Selection and coverage metadata, kept alongside external raw schema pairs
/// so sparse evidence cannot be mistaken for a representative benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ExternalTransitionManifestSummary {
    pub(crate) total_transitions: usize,
    pub(crate) mutation_id_counts: BTreeMap<String, usize>,
    pub(crate) limitations: Vec<String>,
    pub(crate) preserved_source_files: Vec<ExternalPreservedSourceFile>,
    pub(crate) traceability_only: ExternalTraceabilityMetadata,
}

/// Manifest-native audit context that accompanies, and is carried into, every
/// external evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExternalTraceabilityMetadata {
    pub status: String,
    pub accuracy_evidence: bool,
    pub clean_record_count: usize,
    pub clean_record_ids: Vec<String>,
    pub rejected_record_count: usize,
    pub rejection_skew: BTreeMap<String, ExternalRejectionSkew>,
    pub overlap_characterization: String,
    pub preflight_report_path: String,
    pub original_preflight_report_path: String,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExternalRejectionSkew {
    pub selected: usize,
    pub clean: usize,
    pub rejected: usize,
    pub rejection_rate: Option<f64>,
}

/// Immutable OpenAPI source file retained beside the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ExternalPreservedSourceFile {
    pub(crate) revision: String,
    pub(crate) commit: String,
    pub(crate) blob_sha1: String,
    pub(crate) source_path: String,
    pub(crate) audit_copy_path: String,
}

/// Raw schema pairs are supplied only in the local manifest. They are never
/// copied to the resulting report, which retains IDs, hashes, oracle evidence,
/// and predictions instead.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ExternalTransitionInput {
    record_id: String,
    source: String,
    source_kind: ExternalSourceKind,
    source_url: String,
    base_revision: String,
    candidate_revision: String,
    base_commit: String,
    candidate_commit: String,
    base_blob_sha1: String,
    candidate_blob_sha1: String,
    source_license: String,
    source_path: String,
    family_id: String,
    contract_id: String,
    old_version: String,
    new_version: String,
    policy_pack: String,
    mutation_id: String,
    external_use: ExternalUse,
    base_schema: Value,
    candidate_schema: Value,
}

/// Labels and scores external transitions without updating a model, scaler, or
/// prepared training dataset.
pub fn evaluate_external_transitions(
    config: &ExternalEvaluationConfig,
) -> Result<ExternalEvaluationReport, String> {
    let manifest_bytes = fs::read(&config.manifest_path).map_err(|error| {
        format!(
            "could not read external manifest {}: {error}",
            config.manifest_path.display()
        )
    })?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest = serde_json::from_slice::<ExternalTransitionManifest>(&manifest_bytes)
        .map_err(|error| format!("invalid external transition manifest: {error}"))?;
    validate_manifest(&manifest)?;

    let v9_dataset = PreparedDcgDataset::load(&config.v9_dataset_path)
        .map_err(|error| format!("could not load V9 dataset: {error}"))?;
    if v9_dataset.feature_version() != DCG_FEATURE_V6_VERSION {
        return Err(format!(
            "external evaluation requires a {DCG_FEATURE_V6_VERSION} V9 dataset, got {}",
            v9_dataset.feature_version()
        ));
    }
    let v9_dataset_sha256 = sha256_file(&config.v9_dataset_path)?;
    let model_sha256 = sha256_file(&config.model_path)?;
    let model = ThreeWayModelArtifact::load(&config.model_path)
        .map_err(|error| format!("could not load frozen three-way model: {error}"))?;
    verify_frozen_model(&model, &v9_dataset_sha256)?;

    let oracle = PinnedOracle::new(config.oracle_config.clone())
        .map_err(|error| format!("could not initialize pinned oracle: {error}"))?;
    verify_v9_oracle_identity(&v9_dataset, &model, &oracle)?;

    let v9_sources = v9_dataset
        .records()
        .iter()
        .map(|record| record.source.as_str())
        .collect::<BTreeSet<_>>();
    let v9_families = v9_dataset
        .records()
        .iter()
        .map(|record| record.family_id.as_str())
        .collect::<BTreeSet<_>>();
    let v9_pairs = v9_dataset
        .records()
        .iter()
        .filter_map(|record| record.generation.as_ref())
        .map(|generation| generation.pair_fingerprint.as_str())
        .filter(|fingerprint| !fingerprint.is_empty())
        .collect::<BTreeSet<_>>();
    let v9_full_features = v9_dataset
        .records()
        .iter()
        .map(|record| full_feature_signature(&record.features))
        .collect::<HashSet<_>>();
    let v9_structural_features = v9_dataset
        .records()
        .iter()
        .map(|record| policy_free_structural_signature(&record.features))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if v9_structural_features.is_empty() {
        return Err("V9 dataset contains no structural feature signatures".to_owned());
    }

    let v9_policies = v9_dataset
        .records()
        .iter()
        .map(|record| record.policy_pack.as_str())
        .collect::<BTreeSet<_>>();
    let manifest_policies = manifest
        .transitions
        .iter()
        .map(|transition| transition.policy_pack.clone())
        .collect::<BTreeSet<_>>();
    let unsupported_policies = manifest_policies
        .iter()
        .filter(|policy| !v9_policies.contains(policy.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !unsupported_policies.is_empty() {
        return Err(format!(
            "external manifest names policy packs absent from V9: {unsupported_policies:?}; evaluate a separately conformed policy scope instead"
        ));
    }
    let contexts = ApprovedPolicyContexts::load(
        oracle.policy_packs_path(),
        &manifest_policies.into_iter().collect::<Vec<_>>(),
    )
    .map_err(|error| format!("could not resolve external policy contexts: {error}"))?;
    let extractor = SchemaChangeFeatureV6Extractor::new();
    let mut external_pairs = HashSet::new();
    let mut metrics = AccuracyMetrics::default();
    let mut records = Vec::with_capacity(manifest.transitions.len());

    for transition in manifest.transitions {
        // A traceability or preflight-rejected transition is intentionally not
        // labelled, feature-extracted, inferred, or counted. Keeping this
        // decision ahead of oracle setup makes the exclusion mechanically
        // enforceable rather than a reporting convention.
        if !transition.external_use.is_accuracy_bearing() {
            let status = match transition.external_use {
                ExternalUse::TraceabilityOnly => "excluded_traceability",
                ExternalUse::RejectedV9Overlap => "excluded_manifest_overlap",
                ExternalUse::AccuracyBearing => unreachable!("checked above"),
            }
            .to_owned();
            records.push(excluded_transition_record(transition, status));
            continue;
        }
        let base_schema = serde_json::to_string(&transition.base_schema).map_err(|error| {
            format!(
                "could not serialize {} base schema: {error}",
                transition.record_id
            )
        })?;
        let candidate_schema =
            serde_json::to_string(&transition.candidate_schema).map_err(|error| {
                format!(
                    "could not serialize {} candidate schema: {error}",
                    transition.record_id
                )
            })?;
        let pair = GeneratedPair {
            contract_id: transition.contract_id.clone(),
            policy_pack: transition.policy_pack.clone(),
            base_schema: base_schema.clone(),
            candidate_schema: candidate_schema.clone(),
        };
        let run = oracle
            .check_backward(&config.oracle_workspace, &pair)
            .map_err(|error| {
                format!("pinned oracle failed for {}: {error}", transition.record_id)
            })?;
        let pair_fingerprint = canonical_pair_fingerprint(
            &base_schema,
            &candidate_schema,
            &transition.policy_pack,
            oracle.jar_sha256(),
        )
        .map_err(|error| format!("could not fingerprint {}: {error}", transition.record_id))?;
        let actual_label = oracle_label(run.outcome);
        let oracle_evidence = ExternalOracleEvidence {
            outcome: run.outcome.as_str().to_owned(),
            exit_code: run.exit_code,
            stdout: run.stdout,
            stderr: run.stderr,
        };

        // An invalid schema or guarded oracle failure has no trusted label and
        // therefore must never be passed through feature extraction or model
        // inference. Preserve its JAR evidence and continue with the next
        // independently supplied transition instead of failing a whole batch.
        if actual_label.is_none() {
            records.push(ExternalTransitionEvaluationRecord {
                record_id: transition.record_id,
                source: transition.source,
                source_kind: transition.source_kind,
                source_url: transition.source_url,
                base_revision: transition.base_revision,
                candidate_revision: transition.candidate_revision,
                base_commit: transition.base_commit,
                candidate_commit: transition.candidate_commit,
                base_blob_sha1: transition.base_blob_sha1,
                candidate_blob_sha1: transition.candidate_blob_sha1,
                source_license: transition.source_license,
                source_path: transition.source_path,
                family_id: transition.family_id,
                contract_id: transition.contract_id,
                old_version: transition.old_version,
                new_version: transition.new_version,
                policy_pack: transition.policy_pack,
                mutation_id: transition.mutation_id,
                external_use: transition.external_use,
                status: "oracle_rejected".to_owned(),
                rejection_reasons: Vec::new(),
                pair_fingerprint,
                exact_model_feature_overlap: false,
                nearest_policy_free_structural_distance: None,
                oracle: oracle_evidence,
                actual_label: None,
                predicted_label: None,
                predicted_probabilities: None,
            });
            continue;
        }

        let context = contexts.get(&transition.policy_pack).map_err(|error| {
            format!(
                "could not resolve {} policy context: {error}",
                transition.record_id
            )
        })?;
        let features = extractor
            .extract(&base_schema, &candidate_schema, context)
            .map_err(|error| {
                format!(
                    "could not extract {} V6 features: {error}",
                    transition.record_id
                )
            })?;
        let structural_signature = policy_free_structural_signature(&features)?;
        let nearest_distance =
            nearest_structural_distance(&structural_signature, &v9_structural_features)
                .ok_or_else(|| "V9 structural signature index is unexpectedly empty".to_owned())?;
        let exact_model_feature_overlap =
            v9_full_features.contains(&full_feature_signature(&features));
        let mut rejection_reasons = Vec::new();
        if v9_sources.contains(transition.source.as_str()) {
            rejection_reasons.push("source identifier is already present in V9".to_owned());
        }
        if v9_families.contains(transition.family_id.as_str()) {
            rejection_reasons.push("family identifier is already present in V9".to_owned());
        }
        if v9_pairs.contains(pair_fingerprint.as_str()) {
            rejection_reasons
                .push("canonical oracle pair fingerprint is already present in V9".to_owned());
        }
        if exact_model_feature_overlap {
            rejection_reasons
                .push("full V6 model-input feature vector is already present in V9".to_owned());
        }
        if !external_pairs.insert(pair_fingerprint.clone()) {
            rejection_reasons.push(
                "canonical oracle pair fingerprint is duplicated within the external manifest"
                    .to_owned(),
            );
        }
        if nearest_distance <= config.max_structural_coordinate_distance {
            rejection_reasons.push(format!(
                "nearest V9 policy-free structural feature distance {nearest_distance} is within configured rejection threshold {}",
                config.max_structural_coordinate_distance
            ));
        }

        let mut status = "scored".to_owned();
        let mut predicted_label = None;
        let mut predicted_probabilities = None;
        if !rejection_reasons.is_empty() {
            status = "rejected_overlap".to_owned();
        } else {
            let actual =
                actual_label.expect("oracle-accepted transition has a compatibility label");
            let normalized = model
                .scaler()
                .transform_vector(&features)
                .map_err(|error| {
                    format!(
                        "could not apply frozen scaler to {}: {error}",
                        transition.record_id
                    )
                })?;
            let prediction = model.model().predict(&normalized).map_err(|error| {
                format!(
                    "frozen model could not score {}: {error}",
                    transition.record_id
                )
            })?;
            let predicted = prediction.label;
            predicted_label = Some(predicted.as_str().to_owned());
            predicted_probabilities = Some(prediction.probabilities.iter().copied().collect());
            record_accuracy_metrics(
                transition.external_use,
                &transition.mutation_id,
                &transition.policy_pack,
                actual,
                predicted,
                &mut metrics,
            );
        }
        records.push(ExternalTransitionEvaluationRecord {
            record_id: transition.record_id,
            source: transition.source,
            source_kind: transition.source_kind,
            source_url: transition.source_url,
            base_revision: transition.base_revision,
            candidate_revision: transition.candidate_revision,
            base_commit: transition.base_commit,
            candidate_commit: transition.candidate_commit,
            base_blob_sha1: transition.base_blob_sha1,
            candidate_blob_sha1: transition.candidate_blob_sha1,
            source_license: transition.source_license,
            source_path: transition.source_path,
            family_id: transition.family_id,
            contract_id: transition.contract_id,
            old_version: transition.old_version,
            new_version: transition.new_version,
            policy_pack: transition.policy_pack,
            mutation_id: transition.mutation_id,
            external_use: transition.external_use,
            status,
            rejection_reasons,
            pair_fingerprint,
            exact_model_feature_overlap,
            nearest_policy_free_structural_distance: Some(nearest_distance),
            oracle: oracle_evidence,
            actual_label: actual_label.map(|label| label.as_str().to_owned()),
            predicted_label,
            predicted_probabilities,
        });
    }

    let scored_records = records
        .iter()
        .filter(|record| record.status == "scored")
        .count();
    let traceability_excluded_records = records
        .iter()
        .filter(|record| record.status == "excluded_traceability")
        .count();
    let manifest_overlap_excluded_records = records
        .iter()
        .filter(|record| record.status == "excluded_manifest_overlap")
        .count();
    let oracle_rejected_records = records
        .iter()
        .filter(|record| record.status == "oracle_rejected")
        .count();
    let overlap_rejected_records = records
        .iter()
        .filter(|record| record.status == "rejected_overlap")
        .count();
    Ok(ExternalEvaluationReport {
        format_version: EXTERNAL_EVALUATION_REPORT_FORMAT_VERSION.to_owned(),
        mode: "BACKWARD".to_owned(),
        frozen_inference_only: true,
        max_structural_coordinate_distance: config.max_structural_coordinate_distance,
        v9_reference: ExternalV9Reference {
            v9_dataset_path: config.v9_dataset_path.to_string_lossy().into_owned(),
            v9_dataset_sha256,
            model_path: config.model_path.to_string_lossy().into_owned(),
            model_sha256,
            model_version: model.model_version().to_owned(),
            feature_version: model.feature_version().to_owned(),
            oracle_jar_sha256: oracle.jar_sha256().to_owned(),
            policy_packs_sha256: oracle.policy_packs_sha256().to_owned(),
        },
        manifest_sha256,
        traceability: manifest.summary.traceability_only,
        total_manifest_records: records.len(),
        scored_records,
        traceability_excluded_records,
        manifest_overlap_excluded_records,
        oracle_rejected_records,
        overlap_rejected_records,
        overall: metrics.overall.finish(),
        by_mutation: finish_strata(metrics.by_mutation),
        by_policy_pack: finish_strata(metrics.by_policy_pack),
        by_mutation_policy_pack: finish_strata(metrics.by_mutation_policy_pack),
        records,
    })
}

/// Validates an external manifest against V9 without invoking Java, applying
/// a scaler, or running a model. It is the required gate before oracle labels
/// are acquired for genuinely external evidence.
pub fn preflight_external_manifest(
    config: &ExternalManifestPreflightConfig,
) -> Result<ExternalManifestPreflightReport, String> {
    let manifest_bytes = fs::read(&config.manifest_path).map_err(|error| {
        format!(
            "could not read external manifest {}: {error}",
            config.manifest_path.display()
        )
    })?;
    let manifest_sha256 = sha256_bytes(&manifest_bytes);
    let manifest = serde_json::from_slice::<ExternalTransitionManifest>(&manifest_bytes)
        .map_err(|error| format!("invalid external transition manifest: {error}"))?;
    validate_manifest(&manifest)?;
    let v9_dataset = PreparedDcgDataset::load(&config.v9_dataset_path)
        .map_err(|error| format!("could not load V9 dataset: {error}"))?;
    if v9_dataset.feature_version() != DCG_FEATURE_V6_VERSION {
        return Err(format!(
            "external manifest preflight requires a {DCG_FEATURE_V6_VERSION} V9 dataset, got {}",
            v9_dataset.feature_version()
        ));
    }
    let jar_sha256 = sha256_file(&config.jar_path)?;
    let policy_packs_sha256 = sha256_file(&config.policy_packs_path)?;
    let v9_dataset_sha256 = sha256_file(&config.v9_dataset_path)?;
    let v9_sources = v9_dataset
        .records()
        .iter()
        .map(|record| record.source.as_str())
        .collect::<BTreeSet<_>>();
    let v9_families = v9_dataset
        .records()
        .iter()
        .map(|record| record.family_id.as_str())
        .collect::<BTreeSet<_>>();
    let v9_pairs = v9_dataset
        .records()
        .iter()
        .filter_map(|record| record.generation.as_ref())
        .map(|generation| generation.pair_fingerprint.as_str())
        .filter(|fingerprint| !fingerprint.is_empty())
        .collect::<BTreeSet<_>>();
    let v9_full_features = v9_dataset
        .records()
        .iter()
        .map(|record| full_feature_signature(&record.features))
        .collect::<HashSet<_>>();
    let v9_structural_features = v9_dataset
        .records()
        .iter()
        .map(|record| policy_free_structural_signature(&record.features))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let policies = manifest
        .transitions
        .iter()
        .map(|transition| transition.policy_pack.clone())
        .collect::<BTreeSet<_>>();
    let v9_policies = v9_dataset
        .records()
        .iter()
        .map(|record| record.policy_pack.as_str())
        .collect::<BTreeSet<_>>();
    let missing_policies = policies
        .iter()
        .filter(|policy| !v9_policies.contains(policy.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !missing_policies.is_empty() {
        return Err(format!(
            "external manifest names policy packs absent from V9: {missing_policies:?}"
        ));
    }
    let contexts = ApprovedPolicyContexts::load(
        &config.policy_packs_path,
        &policies.into_iter().collect::<Vec<_>>(),
    )
    .map_err(|error| format!("could not resolve external policy contexts: {error}"))?;
    let extractor = SchemaChangeFeatureV6Extractor::new();
    let mut external_pairs = HashSet::new();
    let mut records = Vec::with_capacity(manifest.transitions.len());
    for transition in &manifest.transitions {
        let base_schema = serde_json::to_string(&transition.base_schema).map_err(|error| {
            format!(
                "could not serialize {} base schema: {error}",
                transition.record_id
            )
        })?;
        let candidate_schema =
            serde_json::to_string(&transition.candidate_schema).map_err(|error| {
                format!(
                    "could not serialize {} candidate schema: {error}",
                    transition.record_id
                )
            })?;
        let pair_fingerprint = canonical_pair_fingerprint(
            &base_schema,
            &candidate_schema,
            &transition.policy_pack,
            &jar_sha256,
        )
        .map_err(|error| format!("could not fingerprint {}: {error}", transition.record_id))?;
        let features = extractor
            .extract(
                &base_schema,
                &candidate_schema,
                contexts.get(&transition.policy_pack).map_err(|error| {
                    format!(
                        "could not resolve {} policy context: {error}",
                        transition.record_id
                    )
                })?,
            )
            .map_err(|error| {
                format!(
                    "could not extract {} V6 features: {error}",
                    transition.record_id
                )
            })?;
        let structural = policy_free_structural_signature(&features)?;
        let nearest_distance = nearest_structural_distance(&structural, &v9_structural_features)
            .ok_or_else(|| "V9 structural signature index is unexpectedly empty".to_owned())?;
        let exact_model_feature_overlap =
            v9_full_features.contains(&full_feature_signature(&features));
        let mut rejection_reasons = Vec::new();
        if v9_sources.contains(transition.source.as_str()) {
            rejection_reasons.push("source identifier is already present in V9".to_owned());
        }
        if v9_families.contains(transition.family_id.as_str()) {
            rejection_reasons.push("family identifier is already present in V9".to_owned());
        }
        if v9_pairs.contains(pair_fingerprint.as_str()) {
            rejection_reasons
                .push("canonical pair fingerprint is already present in V9".to_owned());
        }
        if !external_pairs.insert(pair_fingerprint.clone()) {
            rejection_reasons
                .push("canonical pair fingerprint repeats in external manifest".to_owned());
        }
        if exact_model_feature_overlap {
            rejection_reasons
                .push("full V6 model-input feature vector is already present in V9".to_owned());
        }
        if nearest_distance <= config.max_structural_coordinate_distance {
            rejection_reasons.push(format!(
                "nearest V9 policy-free structural feature distance {nearest_distance} is within configured rejection threshold {}",
                config.max_structural_coordinate_distance
            ));
        }
        records.push(ExternalManifestPreflightRecord {
            record_id: transition.record_id.clone(),
            source: transition.source.clone(),
            family_id: transition.family_id.clone(),
            policy_pack: transition.policy_pack.clone(),
            mutation_id: transition.mutation_id.clone(),
            pair_fingerprint,
            exact_model_feature_overlap,
            nearest_policy_free_structural_distance: nearest_distance,
            accepted: rejection_reasons.is_empty(),
            rejection_reasons,
        });
    }
    let accepted_records = records.iter().filter(|record| record.accepted).count();
    Ok(ExternalManifestPreflightReport {
        format_version: "dcg-external-manifest-preflight-report-v1".to_owned(),
        oracle_invoked: false,
        model_inference_invoked: false,
        manifest_sha256,
        v9_dataset_sha256,
        oracle_jar_sha256: jar_sha256,
        policy_packs_sha256,
        max_structural_coordinate_distance: config.max_structural_coordinate_distance,
        passed: accepted_records == records.len(),
        total_records: records.len(),
        accepted_records,
        rejected_records: records.len() - accepted_records,
        records,
    })
}

/// Explains one external transition's label-free V6 feature coordinates and
/// the closest policy-free structural vectors already present in V9. Unlike a
/// preflight, this does not need an oracle JAR because it does not calculate a
/// pair fingerprint or invoke the compatibility executable.
pub fn diagnose_external_feature_space(
    config: &ExternalFeatureDiagnosticConfig,
) -> Result<ExternalFeatureDiagnosticReport, String> {
    if config.nearest_record_limit == 0 {
        return Err("nearest_record_limit must be at least one".to_owned());
    }
    let manifest_bytes = fs::read(&config.manifest_path).map_err(|error| {
        format!(
            "could not read external manifest {}: {error}",
            config.manifest_path.display()
        )
    })?;
    let manifest = serde_json::from_slice::<ExternalTransitionManifest>(&manifest_bytes)
        .map_err(|error| format!("invalid external transition manifest: {error}"))?;
    validate_manifest(&manifest)?;
    let transition = manifest
        .transitions
        .iter()
        .find(|transition| transition.record_id == config.record_id)
        .ok_or_else(|| format!("external manifest does not contain `{}`", config.record_id))?;
    let v9_dataset = PreparedDcgDataset::load(&config.v9_dataset_path)
        .map_err(|error| format!("could not load V9 dataset: {error}"))?;
    if v9_dataset.feature_version() != DCG_FEATURE_V6_VERSION {
        return Err(format!(
            "external feature diagnostic requires a {DCG_FEATURE_V6_VERSION} V9 dataset, got {}",
            v9_dataset.feature_version()
        ));
    }
    let contexts = ApprovedPolicyContexts::load(
        &config.policy_packs_path,
        std::slice::from_ref(&transition.policy_pack),
    )
    .map_err(|error| format!("could not resolve external policy context: {error}"))?;
    let base_schema = serde_json::to_string(&transition.base_schema).map_err(|error| {
        format!(
            "could not serialize {} base schema: {error}",
            transition.record_id
        )
    })?;
    let candidate_schema =
        serde_json::to_string(&transition.candidate_schema).map_err(|error| {
            format!(
                "could not serialize {} candidate schema: {error}",
                transition.record_id
            )
        })?;
    let features = SchemaChangeFeatureV6Extractor::new()
        .extract(
            &base_schema,
            &candidate_schema,
            contexts.get(&transition.policy_pack).map_err(|error| {
                format!(
                    "could not resolve {} policy context: {error}",
                    transition.record_id
                )
            })?,
        )
        .map_err(|error| {
            format!(
                "could not extract {} V6 features: {error}",
                transition.record_id
            )
        })?;
    let structural_indices = policy_free_structural_indices();
    let structural_coordinates = feature_coordinates(&features, &structural_indices);
    let nonzero_v6_features = features
        .iter()
        .enumerate()
        .filter(|(_, value)| **value != 0.0)
        .map(|(index, value)| ExternalFeatureCoordinate {
            index,
            name: DCG_FEATURE_V6_NAMES[index].to_owned(),
            value: *value,
        })
        .collect::<Vec<_>>();
    let full_signature = full_feature_signature(&features);
    let exact_full_v6_feature_match_count = v9_dataset
        .records()
        .iter()
        .filter(|record| full_feature_signature(&record.features) == full_signature)
        .count();

    let mut nearest_distance = None;
    let mut nearest = Vec::new();
    for record in v9_dataset.records() {
        let distance = structural_coordinate_distance(&features, &record.features)?;
        match nearest_distance {
            None => {
                nearest_distance = Some(distance);
                nearest.push(record);
            }
            Some(current) if distance < current => {
                nearest_distance = Some(distance);
                nearest.clear();
                nearest.push(record);
            }
            Some(current) if distance == current => nearest.push(record),
            Some(_) => {}
        }
    }
    let nearest_distance = nearest_distance
        .ok_or_else(|| "V9 dataset is unexpectedly empty during feature diagnostic".to_owned())?;
    nearest.sort_by(|left, right| left.record_id.cmp(&right.record_id));
    let nearest_v9_record_count = nearest.len();
    let nearest_v9_records = nearest
        .into_iter()
        .take(config.nearest_record_limit)
        .map(|record| ExternalFeatureDiagnosticNeighbor {
            record_id: record.record_id.clone(),
            source: record.source.clone(),
            family_id: record.family_id.clone(),
            policy_pack: record.policy_pack.clone(),
            declared_mutation: record
                .generation
                .as_ref()
                .map(|generation| generation.declared_mutation.clone()),
            mutation_variant: record
                .generation
                .as_ref()
                .map(|generation| generation.mutation_variant.clone()),
            pair_fingerprint: record
                .generation
                .as_ref()
                .map(|generation| generation.pair_fingerprint.clone()),
            structural_coordinate_distance: nearest_distance,
            differing_structural_coordinates: structural_coordinate_differences(
                &features,
                &record.features,
            ),
        })
        .collect::<Vec<_>>();

    Ok(ExternalFeatureDiagnosticReport {
        format_version: "dcg-external-feature-diagnostic-v1".to_owned(),
        oracle_invoked: false,
        model_inference_invoked: false,
        labels_produced: false,
        record_id: transition.record_id.clone(),
        source: transition.source.clone(),
        source_path: transition.source_path.clone(),
        mutation_id: transition.mutation_id.clone(),
        policy_pack: transition.policy_pack.clone(),
        full_v6_feature_count: features.len(),
        structural_coordinate_count: structural_coordinates.len(),
        structural_coordinates,
        nonzero_v6_features,
        exact_full_v6_feature_match_count,
        nearest_v9_structural_distance: nearest_distance,
        nearest_v9_record_count,
        nearest_v9_records,
    })
}

fn validate_manifest(manifest: &ExternalTransitionManifest) -> Result<(), String> {
    if manifest.format_version != EXTERNAL_TRANSITION_MANIFEST_FORMAT_VERSION {
        return Err(format!(
            "external manifest format must be {EXTERNAL_TRANSITION_MANIFEST_FORMAT_VERSION}, got {}",
            manifest.format_version
        ));
    }
    if manifest.transitions.is_empty() {
        return Err("external manifest must contain at least one transition".to_owned());
    }
    if manifest.summary.total_transitions != manifest.transitions.len() {
        return Err(format!(
            "external manifest summary total_transitions={} does not match {} records",
            manifest.summary.total_transitions,
            manifest.transitions.len()
        ));
    }
    let actual_mutation_counts = manifest.transitions.iter().fold(
        BTreeMap::<String, usize>::new(),
        |mut counts, transition| {
            *counts.entry(transition.mutation_id.clone()).or_default() += 1;
            counts
        },
    );
    for (mutation_id, actual_count) in &actual_mutation_counts {
        if manifest.summary.mutation_id_counts.get(mutation_id) != Some(actual_count) {
            return Err(format!(
                "external manifest mutation_id_counts does not match records for {mutation_id}"
            ));
        }
    }
    if manifest
        .summary
        .mutation_id_counts
        .iter()
        .any(|(mutation_id, count)| *count > 0 && !actual_mutation_counts.contains_key(mutation_id))
    {
        return Err(
            "external manifest has a positive summary count without a matching record".to_owned(),
        );
    }
    if manifest
        .summary
        .limitations
        .iter()
        .any(|value| value.trim().is_empty())
    {
        return Err("external manifest limitations must not contain empty entries".to_owned());
    }
    if manifest.summary.preserved_source_files.is_empty() {
        return Err("external manifest must name its preserved source files".to_owned());
    }
    let traceability = &manifest.summary.traceability_only;
    if traceability.status != "traceability-only" {
        return Err("external manifest traceability status must be traceability-only".to_owned());
    }
    if traceability.accuracy_evidence {
        return Err(
            "external manifest traceability metadata cannot claim accuracy evidence".to_owned(),
        );
    }
    if traceability.clean_record_count != traceability.clean_record_ids.len() {
        return Err(
            "external manifest traceability clean_record_count does not match clean_record_ids"
                .to_owned(),
        );
    }
    if traceability.overlap_characterization.trim().is_empty() {
        return Err(
            "external manifest traceability overlap_characterization must not be empty".to_owned(),
        );
    }
    if traceability
        .rejection_skew
        .values()
        .any(|skew| skew.selected != skew.clean + skew.rejected)
    {
        return Err(
            "external manifest traceability rejection_skew totals are inconsistent".to_owned(),
        );
    }
    let mut record_ids = BTreeSet::new();
    let mut traceability_ids = BTreeSet::new();
    for transition in &manifest.transitions {
        for (field, value) in [
            ("record_id", transition.record_id.as_str()),
            ("source", transition.source.as_str()),
            ("source_url", transition.source_url.as_str()),
            ("base_revision", transition.base_revision.as_str()),
            ("candidate_revision", transition.candidate_revision.as_str()),
            ("base_commit", transition.base_commit.as_str()),
            ("candidate_commit", transition.candidate_commit.as_str()),
            ("base_blob_sha1", transition.base_blob_sha1.as_str()),
            (
                "candidate_blob_sha1",
                transition.candidate_blob_sha1.as_str(),
            ),
            ("source_license", transition.source_license.as_str()),
            ("source_path", transition.source_path.as_str()),
            ("family_id", transition.family_id.as_str()),
            ("contract_id", transition.contract_id.as_str()),
            ("old_version", transition.old_version.as_str()),
            ("new_version", transition.new_version.as_str()),
            ("policy_pack", transition.policy_pack.as_str()),
            ("mutation_id", transition.mutation_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(format!(
                    "external transition `{}` has an empty {field}",
                    transition.record_id
                ));
            }
        }
        match transition.source_kind {
            ExternalSourceKind::PublicVersionHistory
            | ExternalSourceKind::PublicSchemaRepository
                if !is_public_source_url(&transition.source_url) =>
            {
                return Err(format!(
                    "external transition `{}` requires an http(s) source_url for a public source",
                    transition.record_id
                ));
            }
            ExternalSourceKind::ManualUnrelated
                if !transition.source_url.starts_with("manual://") =>
            {
                return Err(format!(
                    "external transition `{}` requires a manual:// source_url for a manually authored unrelated source",
                    transition.record_id
                ));
            }
            _ => {}
        }
        if !record_ids.insert(transition.record_id.as_str()) {
            return Err(format!(
                "external manifest repeats record_id `{}`",
                transition.record_id
            ));
        }
        if transition.external_use == ExternalUse::TraceabilityOnly
            && !traceability_ids.insert(transition.record_id.as_str())
        {
            return Err(format!(
                "external manifest repeats traceability record_id `{}`",
                transition.record_id
            ));
        }
    }
    let declared_traceability_ids = traceability
        .clean_record_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if traceability_ids != declared_traceability_ids {
        return Err(
            "external manifest traceability clean_record_ids do not exactly match traceability-only transitions"
                .to_owned(),
        );
    }
    Ok(())
}

fn is_public_source_url(value: &str) -> bool {
    value.starts_with("https://") || value.starts_with("http://")
}

fn verify_frozen_model(
    model: &ThreeWayModelArtifact,
    v9_dataset_sha256: &str,
) -> Result<(), String> {
    if model.feature_version() != DCG_FEATURE_V6_VERSION {
        return Err(format!(
            "frozen model requires {}, got {}",
            DCG_FEATURE_V6_VERSION,
            model.feature_version()
        ));
    }
    if !model.model_version().contains("normal-family-split") {
        return Err(format!(
            "external evaluation requires a frozen V9 normal-family-split model, got {}",
            model.model_version()
        ));
    }
    if model.input_provenance().dataset_sha256 != v9_dataset_sha256 {
        return Err("frozen model dataset SHA-256 does not match supplied V9 dataset".to_owned());
    }
    Ok(())
}

fn verify_v9_oracle_identity(
    v9_dataset: &PreparedDcgDataset,
    model: &ThreeWayModelArtifact,
    oracle: &PinnedOracle,
) -> Result<(), String> {
    if model.input_provenance().oracle_jar_sha256 != oracle.jar_sha256() {
        return Err("frozen model JAR SHA-256 differs from supplied oracle JAR".to_owned());
    }
    if model.input_provenance().policy_packs_sha256 != oracle.policy_packs_sha256() {
        return Err(
            "frozen model policy-pack SHA-256 differs from supplied policy-pack file".to_owned(),
        );
    }
    let dataset_jar_hashes = v9_dataset
        .records()
        .iter()
        .filter_map(|record| record.generation.as_ref())
        .map(|generation| generation.oracle_jar_sha256.as_str())
        .collect::<BTreeSet<_>>();
    let dataset_policy_hashes = v9_dataset
        .records()
        .iter()
        .filter_map(|record| record.generation.as_ref())
        .map(|generation| generation.policy_packs_sha256.as_str())
        .collect::<BTreeSet<_>>();
    if dataset_jar_hashes != BTreeSet::from([oracle.jar_sha256()])
        || dataset_policy_hashes != BTreeSet::from([oracle.policy_packs_sha256()])
    {
        return Err("supplied V9 dataset does not share the frozen oracle identities".to_owned());
    }
    Ok(())
}

fn oracle_label(outcome: OracleOutcome) -> Option<CompatibilityLabel> {
    match outcome {
        OracleOutcome::Safe => Some(CompatibilityLabel::Safe),
        OracleOutcome::Warning => Some(CompatibilityLabel::Warning),
        OracleOutcome::Breaking => Some(CompatibilityLabel::Breaking),
        OracleOutcome::Rejected => None,
    }
}

fn excluded_transition_record(
    transition: ExternalTransitionInput,
    status: String,
) -> ExternalTransitionEvaluationRecord {
    ExternalTransitionEvaluationRecord {
        record_id: transition.record_id,
        source: transition.source,
        source_kind: transition.source_kind,
        source_url: transition.source_url,
        base_revision: transition.base_revision,
        candidate_revision: transition.candidate_revision,
        base_commit: transition.base_commit,
        candidate_commit: transition.candidate_commit,
        base_blob_sha1: transition.base_blob_sha1,
        candidate_blob_sha1: transition.candidate_blob_sha1,
        source_license: transition.source_license,
        source_path: transition.source_path,
        family_id: transition.family_id,
        contract_id: transition.contract_id,
        old_version: transition.old_version,
        new_version: transition.new_version,
        policy_pack: transition.policy_pack,
        mutation_id: transition.mutation_id,
        external_use: transition.external_use,
        status,
        rejection_reasons: Vec::new(),
        pair_fingerprint: String::new(),
        exact_model_feature_overlap: false,
        nearest_policy_free_structural_distance: None,
        oracle: ExternalOracleEvidence {
            outcome: "not-invoked".to_owned(),
            exit_code: 0,
            stdout: String::new(),
            stderr: String::new(),
        },
        actual_label: None,
        predicted_label: None,
        predicted_probabilities: None,
    }
}

fn full_feature_signature(features: &Vector) -> Vec<u64> {
    features.iter().map(|value| value.to_bits()).collect()
}

/// Returns V6's 49 label-free structural coordinates, excluding resolved
/// policy context (24..48) and active policy action counts (59..62).
fn policy_free_structural_signature(features: &Vector) -> Result<Vec<u64>, String> {
    if features.len() != 76 {
        return Err(format!(
            "expected 76 V6 features for overlap audit, got {}",
            features.len()
        ));
    }
    Ok(features
        .iter()
        .enumerate()
        .filter(|(index, _)| is_policy_free_structural_coordinate(*index))
        .map(|(_, value)| value.to_bits())
        .collect())
}

fn policy_free_structural_indices() -> Vec<usize> {
    (0..DCG_FEATURE_V6_NAMES.len())
        .filter(|index| is_policy_free_structural_coordinate(*index))
        .collect()
}

fn is_policy_free_structural_coordinate(index: usize) -> bool {
    !(24..48).contains(&index) && !(59..62).contains(&index)
}

fn feature_coordinates(features: &Vector, indices: &[usize]) -> Vec<ExternalFeatureCoordinate> {
    indices
        .iter()
        .map(|index| ExternalFeatureCoordinate {
            index: *index,
            name: DCG_FEATURE_V6_NAMES[*index].to_owned(),
            value: features[*index],
        })
        .collect()
}

fn structural_coordinate_distance(left: &Vector, right: &Vector) -> Result<usize, String> {
    let left = policy_free_structural_signature(left)?;
    let right = policy_free_structural_signature(right)?;
    Ok(left
        .iter()
        .zip(right)
        .filter(|(left, right)| *left != right)
        .count())
}

fn structural_coordinate_differences(
    external: &Vector,
    v9: &Vector,
) -> Vec<ExternalFeatureCoordinateDifference> {
    policy_free_structural_indices()
        .into_iter()
        .filter(|index| external[*index].to_bits() != v9[*index].to_bits())
        .map(|index| ExternalFeatureCoordinateDifference {
            index,
            name: DCG_FEATURE_V6_NAMES[index].to_owned(),
            external_value: external[index],
            v9_value: v9[index],
        })
        .collect()
}

fn nearest_structural_distance(
    signature: &[u64],
    candidates: &BTreeSet<Vec<u64>>,
) -> Option<usize> {
    candidates
        .iter()
        .map(|candidate| {
            signature
                .iter()
                .zip(candidate)
                .filter(|(left, right)| left != right)
                .count()
        })
        .min()
}

fn sha256_file(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|contents| sha256_bytes(&contents))
        .map_err(|error| format!("could not read {}: {error}", path.display()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Debug, Clone, Default)]
struct MetricAccumulator {
    records: usize,
    correct: usize,
    confusion_matrix: [[usize; 3]; 3],
}

#[derive(Debug, Clone, Default)]
struct AccuracyMetrics {
    overall: MetricAccumulator,
    by_mutation: BTreeMap<String, MetricAccumulator>,
    by_policy_pack: BTreeMap<String, MetricAccumulator>,
    by_mutation_policy_pack: BTreeMap<String, MetricAccumulator>,
}

impl MetricAccumulator {
    fn record(&mut self, actual: CompatibilityLabel, predicted: CompatibilityLabel) {
        self.records += 1;
        self.confusion_matrix[actual.class_index() as usize][predicted.class_index() as usize] += 1;
        if actual == predicted {
            self.correct += 1;
        }
    }

    fn finish(self) -> ExternalMetricReport {
        ExternalMetricReport {
            records: self.records,
            accuracy: (self.records > 0).then(|| self.correct as f64 / self.records as f64),
            confusion_matrix: self.confusion_matrix,
        }
    }
}

/// The sole mutation path for accuracy-bearing metrics. Callers may retain an
/// excluded row in a report, but it cannot change a matrix through this path.
fn record_accuracy_metrics(
    external_use: ExternalUse,
    mutation_id: &str,
    policy_pack: &str,
    actual: CompatibilityLabel,
    predicted: CompatibilityLabel,
    metrics: &mut AccuracyMetrics,
) {
    if !external_use.is_accuracy_bearing() {
        return;
    }
    metrics.overall.record(actual, predicted);
    metrics
        .by_mutation
        .entry(mutation_id.to_owned())
        .or_default()
        .record(actual, predicted);
    metrics
        .by_policy_pack
        .entry(policy_pack.to_owned())
        .or_default()
        .record(actual, predicted);
    metrics
        .by_mutation_policy_pack
        .entry(format!("{mutation_id}|{policy_pack}"))
        .or_default()
        .record(actual, predicted);
}

fn finish_strata(
    values: BTreeMap<String, MetricAccumulator>,
) -> BTreeMap<String, ExternalMetricReport> {
    values
        .into_iter()
        .map(|(key, accumulator)| (key, accumulator.finish()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_traceability_metadata(clean_record_ids: Vec<String>) -> ExternalTraceabilityMetadata {
        ExternalTraceabilityMetadata {
            status: "traceability-only".to_owned(),
            accuracy_evidence: false,
            clean_record_count: clean_record_ids.len(),
            clean_record_ids,
            rejected_record_count: 0,
            rejection_skew: BTreeMap::new(),
            overlap_characterization: "structural comparison only".to_owned(),
            preflight_report_path: "preflight.json".to_owned(),
            original_preflight_report_path: "original-preflight.json".to_owned(),
            limitations: vec!["test limitation".to_owned()],
        }
    }

    #[test]
    fn structural_signature_excludes_policy_context_but_retains_schema_facts() {
        let mut first = vec![0.0; 76];
        first[11] = 1.0;
        first[24 + 4 * 3 + 1] = 1.0;
        first[59] = 1.0;
        let mut second = first.clone();
        second[24 + 4 * 3 + 1] = 0.0;
        second[24 + 4 * 3 + 2] = 1.0;
        second[59] = 0.0;
        second[61] = 1.0;
        assert_eq!(
            policy_free_structural_signature(&Vector::new(first)).unwrap(),
            policy_free_structural_signature(&Vector::new(second)).unwrap()
        );
    }

    #[test]
    fn manifest_requires_independent_identifiers() {
        let manifest = ExternalTransitionManifest {
            format_version: EXTERNAL_TRANSITION_MANIFEST_FORMAT_VERSION.to_owned(),
            summary: ExternalTransitionManifestSummary {
                total_transitions: 1,
                mutation_id_counts: BTreeMap::from([("field_removed".to_owned(), 1)]),
                limitations: vec!["test limitation".to_owned()],
                preserved_source_files: vec![ExternalPreservedSourceFile {
                    revision: "v1.0.0".to_owned(),
                    commit: "a".repeat(40),
                    blob_sha1: "b".repeat(40),
                    source_path: "schema.json".to_owned(),
                    audit_copy_path: "sources/schema.json".to_owned(),
                }],
                traceability_only: test_traceability_metadata(Vec::new()),
            },
            transitions: vec![ExternalTransitionInput {
                record_id: "record-1".to_owned(),
                source: String::new(),
                source_kind: ExternalSourceKind::PublicVersionHistory,
                source_url: "https://example.com/contracts".to_owned(),
                base_revision: "v1.0.0".to_owned(),
                candidate_revision: "v1.0.1".to_owned(),
                base_commit: "a".repeat(40),
                candidate_commit: "b".repeat(40),
                base_blob_sha1: "c".repeat(40),
                candidate_blob_sha1: "d".repeat(40),
                source_license: "Apache-2.0".to_owned(),
                source_path: "schemas/contract.json".to_owned(),
                family_id: "family-1".to_owned(),
                contract_id: "contract-1".to_owned(),
                old_version: "1.0.0".to_owned(),
                new_version: "1.0.1".to_owned(),
                policy_pack: "baseline".to_owned(),
                mutation_id: "field_removed".to_owned(),
                external_use: ExternalUse::AccuracyBearing,
                base_schema: Value::Null,
                candidate_schema: Value::Null,
            }],
        };
        assert!(validate_manifest(&manifest).is_err());
    }

    #[test]
    fn manifest_rejects_missing_public_source_url() {
        let manifest = ExternalTransitionManifest {
            format_version: EXTERNAL_TRANSITION_MANIFEST_FORMAT_VERSION.to_owned(),
            summary: ExternalTransitionManifestSummary {
                total_transitions: 1,
                mutation_id_counts: BTreeMap::from([("field_removed".to_owned(), 1)]),
                limitations: vec!["test limitation".to_owned()],
                preserved_source_files: vec![ExternalPreservedSourceFile {
                    revision: "v1.0.0".to_owned(),
                    commit: "a".repeat(40),
                    blob_sha1: "b".repeat(40),
                    source_path: "schema.json".to_owned(),
                    audit_copy_path: "sources/schema.json".to_owned(),
                }],
                traceability_only: test_traceability_metadata(Vec::new()),
            },
            transitions: vec![ExternalTransitionInput {
                record_id: "record-1".to_owned(),
                source: "external-contract-history".to_owned(),
                source_kind: ExternalSourceKind::PublicVersionHistory,
                source_url: "manual://not-public".to_owned(),
                base_revision: "v1.0.0".to_owned(),
                candidate_revision: "v1.0.1".to_owned(),
                base_commit: "a".repeat(40),
                candidate_commit: "b".repeat(40),
                base_blob_sha1: "c".repeat(40),
                candidate_blob_sha1: "d".repeat(40),
                source_license: "Apache-2.0".to_owned(),
                source_path: "schemas/contract.json".to_owned(),
                family_id: "family-1".to_owned(),
                contract_id: "contract-1".to_owned(),
                old_version: "1.0.0".to_owned(),
                new_version: "1.0.1".to_owned(),
                policy_pack: "baseline".to_owned(),
                mutation_id: "field_removed".to_owned(),
                external_use: ExternalUse::AccuracyBearing,
                base_schema: Value::Null,
                candidate_schema: Value::Null,
            }],
        };
        assert!(validate_manifest(&manifest).is_err());
    }

    #[test]
    fn manifest_rejects_missing_or_unrecognized_external_use() {
        let manifest_json = include_str!(
            "../../data/external/stripe-openapi-v3/stripe-openapi-external-manifest-v3.json"
        );
        let mut missing: Value = serde_json::from_str(manifest_json).unwrap();
        missing["transitions"][0]
            .as_object_mut()
            .unwrap()
            .remove("external_use");
        assert!(serde_json::from_value::<ExternalTransitionManifest>(missing).is_err());

        let mut unknown: Value = serde_json::from_str(manifest_json).unwrap();
        unknown["transitions"][0]["external_use"] = Value::String("not-a-disposition".to_owned());
        assert!(serde_json::from_value::<ExternalTransitionManifest>(unknown).is_err());
    }

    #[test]
    fn dry_aggregation_of_all_stripe_traceability_rows_is_zero() {
        let manifest: ExternalTransitionManifest = serde_json::from_str(include_str!(
            "../../data/external/stripe-openapi-v3/stripe-openapi-external-manifest-v3.json"
        ))
        .unwrap();
        validate_manifest(&manifest).unwrap();
        let traceability_rows = manifest
            .transitions
            .iter()
            .filter(|transition| transition.external_use == ExternalUse::TraceabilityOnly)
            .collect::<Vec<_>>();
        assert_eq!(traceability_rows.len(), 16);

        let mut metrics = AccuracyMetrics::default();
        for transition in traceability_rows {
            // Deliberately stubbed labels: this calls no JAR and no model.
            record_accuracy_metrics(
                transition.external_use,
                &transition.mutation_id,
                &transition.policy_pack,
                CompatibilityLabel::Safe,
                CompatibilityLabel::Breaking,
                &mut metrics,
            );
        }
        let output = serde_json::json!({
            "traceability_rows": 16,
            "overall": metrics.overall.finish(),
            "by_mutation": finish_strata(metrics.by_mutation),
            "by_policy_pack": finish_strata(metrics.by_policy_pack),
            "by_mutation_policy_pack": finish_strata(metrics.by_mutation_policy_pack),
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
        assert_eq!(output["overall"]["records"], 0);
        assert_eq!(
            output["overall"]["confusion_matrix"],
            serde_json::json!([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        );
        assert_eq!(output["by_mutation"], serde_json::json!({}));
        assert_eq!(output["by_policy_pack"], serde_json::json!({}));
        assert_eq!(output["by_mutation_policy_pack"], serde_json::json!({}));
    }

    #[test]
    fn stripe_field_removed_structural_vector_is_available_without_oracle_or_model() {
        let manifest: ExternalTransitionManifest = serde_json::from_str(include_str!(
            "../../data/external/stripe-openapi-v3/stripe-openapi-external-manifest-v3.json"
        ))
        .unwrap();
        let transition = manifest
            .transitions
            .iter()
            .find(|transition| transition.record_id == "stripe.openapi.v2348-to-v2349.card")
            .unwrap();
        let contexts = ApprovedPolicyContexts::from_json(
            r#"{"defaultPack":"baseline","packs":{"baseline":{"rules":{}}}}"#,
            std::slice::from_ref(&transition.policy_pack),
        )
        .unwrap();
        let base = serde_json::to_string(&transition.base_schema).unwrap();
        let candidate = serde_json::to_string(&transition.candidate_schema).unwrap();
        let features = SchemaChangeFeatureV6Extractor::new()
            .extract(
                &base,
                &candidate,
                contexts.get(&transition.policy_pack).unwrap(),
            )
            .unwrap();
        let structural = policy_free_structural_signature(&features).unwrap();
        let output = serde_json::json!({
            "record_id": transition.record_id,
            "mutation_id": transition.mutation_id,
            "structural_coordinate_count": structural.len(),
            "structural_vector": structural,
            "nonzero_v6_features": features.iter().enumerate()
                .filter(|(index, value)| !(24..48).contains(index) && !(59..62).contains(index) && **value != 0.0)
                .map(|(index, value)| serde_json::json!({"index": index, "name": crate::features::DCG_FEATURE_V6_NAMES[index], "value": value}))
                .collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap());
        assert_eq!(structural.len(), 49);
        assert_eq!(features[2], 1.0);
        assert_eq!(features[0], 31.0);
    }
}
