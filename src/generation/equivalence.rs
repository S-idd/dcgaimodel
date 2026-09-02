//! Behavioral replay of the frozen BACKWARD V9 evidence against a distinct
//! reproducible BACKWARD V10 candidate.
//!
//! The audit never rewrites or relabels V9. It reconstructs each preserved
//! schema pair, verifies its historical pair fingerprint, invokes V10 in
//! BACKWARD mode, and compares the new decision with the persisted V9 result.

use super::{
    GeneratedPair, GeneratorError, JsonSchemaBenchSource, OracleConfig, OracleOutcome,
    PinnedOracle, candidates_for_schema, canonical_pair_fingerprint,
};
use crate::models::{OracleCompatibilityMode, PreparedDcgDataset};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::thread;

pub const HISTORICAL_V9_JAR_SHA256: &str =
    "809b25e627e43f847f00a0ce87dcde33ad359006bf22be403e26d662274677dc";
pub const HISTORICAL_V9_DATASET_SHA256: &str =
    "fa09e645480737ba856940778264284d8e832f4532e989c92a259320cfe05ad3";
pub const BACKWARD_V10_JAR_SHA256: &str =
    "c00da951bac88d2be245e08917e6fc62f55abf3a50aad2dea67178a82285f6b6";
pub const PINNED_POLICY_PACKS_SHA256: &str =
    "8f82b058f81ace43c89180803c7ec26ac734b84d0092036a77115688337e1bb6";

#[derive(Debug, Clone)]
pub struct BackwardEquivalenceAuditConfig {
    pub v9_dataset_path: PathBuf,
    pub source_parquet_path: PathBuf,
    pub v10_jar_path: PathBuf,
    pub divergent_control_jar_path: PathBuf,
    pub policy_packs_path: PathBuf,
    pub workspace: PathBuf,
    pub workers: usize,
}

#[derive(Debug, Serialize)]
pub struct BackwardEquivalenceAuditReport {
    pub format_version: &'static str,
    pub status: String,
    pub identity_boundary: IdentityBoundary,
    pub inputs: AuditInputs,
    pub reconstruction: ReconstructionSummary,
    pub sensitivity_control: SensitivityControl,
    pub synthetic_negative_controls: Vec<SyntheticControl>,
    pub preserved_v9_replay: ReplaySummary,
    pub invocations: Vec<InvocationComparison>,
    pub concrete_traces: Vec<ConcreteTrace>,
    pub conclusion: AuditConclusion,
}

#[derive(Debug, Serialize)]
pub struct IdentityBoundary {
    pub historical_v9_jar_sha256: &'static str,
    pub historical_v9_binary_available: bool,
    pub backward_v10_jar_sha256: &'static str,
    pub permanent_rule: &'static str,
}

#[derive(Debug, Serialize)]
pub struct AuditInputs {
    pub v9_dataset_path: String,
    pub v9_dataset_sha256: String,
    pub source_parquet_path: String,
    pub source_parquet_sha256: String,
    pub v10_jar_path: String,
    pub v10_jar_sha256: String,
    pub divergent_control_jar_path: String,
    pub divergent_control_jar_sha256: String,
    pub policy_packs_path: String,
    pub policy_packs_sha256: String,
    pub compatibility_mode: &'static str,
    pub workers: usize,
}

#[derive(Debug, Serialize)]
pub struct ReconstructionSummary {
    pub records: usize,
    pub source_parquet_families_loaded: usize,
    pub v9_families_reconstructed: usize,
    pub generation_seed: u64,
    pub historical_pair_fingerprints_verified: usize,
    pub historical_pair_fingerprint_failures: usize,
}

#[derive(Debug, Serialize)]
pub struct SensitivityControl {
    pub id: &'static str,
    pub shared_code_path: &'static str,
    pub mechanism: &'static str,
    pub v10_result: String,
    pub known_divergent_jar_sha256: String,
    pub known_divergent_result: String,
    pub divergence_detected: bool,
    pub scope_note: &'static str,
}

#[derive(Debug, Serialize)]
pub struct SyntheticControl {
    pub id: String,
    pub purpose: String,
    pub expected_outcome: String,
    pub expected_message_fragment: Option<String>,
    pub actual_outcome: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub rejection_stage: Option<String>,
    pub rejection_reason: Option<String>,
    pub passed: bool,
    pub base_schema_sha256: String,
    pub candidate_schema_sha256: String,
}

#[derive(Debug, Clone)]
struct ReplayJob {
    index: usize,
    record_id: String,
    family_id: String,
    policy_pack: String,
    declared_mutation: String,
    mutation_variant: String,
    pair_fingerprint: String,
    expected_outcome: String,
    expected_stdout: String,
    base_schema: Arc<String>,
    candidate_schema: Arc<String>,
}

#[derive(Debug, Serialize)]
pub struct InvocationComparison {
    pub index: usize,
    pub record_id: String,
    pub family_id: String,
    pub policy_pack: String,
    pub declared_mutation: String,
    pub mutation_variant: String,
    pub historical_pair_fingerprint: String,
    pub expected_v9_outcome: String,
    pub actual_v10_outcome: String,
    pub exit_code: i32,
    pub expected_stdout_sha256: String,
    pub actual_stdout_sha256: String,
    pub outcome_match: bool,
    pub stdout_match: bool,
    pub stderr_empty: bool,
    pub rejection_stage: Option<String>,
    pub rejection_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mismatched_expected_stdout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mismatched_actual_stdout: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ConcreteTrace {
    pub record_id: String,
    pub family_id: String,
    pub policy_pack: String,
    pub declared_mutation: String,
    pub mutation_variant: String,
    pub historical_pair_fingerprint: String,
    pub expected_v9_outcome: String,
    pub actual_v10_outcome: String,
    pub expected_v9_stdout: String,
    pub actual_v10_stdout: String,
}

#[derive(Debug, Serialize)]
pub struct ReplaySummary {
    pub invocations: usize,
    pub accepted: usize,
    pub rejected: usize,
    pub outcome_matches: usize,
    pub outcome_mismatches: usize,
    pub exact_stdout_matches: usize,
    pub exact_stdout_mismatches: usize,
    pub stderr_nonempty: usize,
    pub by_mutation: BTreeMap<String, OutcomeCounts>,
    pub by_policy_pack: BTreeMap<String, OutcomeCounts>,
}

#[derive(Debug, Default, Serialize)]
pub struct OutcomeCounts {
    pub invocations: usize,
    pub safe: usize,
    pub warning: usize,
    pub breaking: usize,
    pub rejected: usize,
    pub outcome_mismatches: usize,
    pub stdout_mismatches: usize,
}

#[derive(Debug, Serialize)]
pub struct AuditConclusion {
    pub behavioral_result: String,
    pub all_preserved_v9_outcomes_match: bool,
    pub all_preserved_v9_stdout_match: bool,
    pub negative_controls_passed: bool,
    pub identity_interchangeable: bool,
    pub allowed_name: &'static str,
    pub prohibited_claim: &'static str,
}

#[derive(Debug, Clone)]
struct SyntheticCase {
    id: &'static str,
    purpose: &'static str,
    base: &'static str,
    candidate: &'static str,
    expected_outcome: &'static str,
    expected_message_fragment: Option<&'static str>,
}

pub fn run_backward_equivalence_audit(
    config: &BackwardEquivalenceAuditConfig,
) -> Result<BackwardEquivalenceAuditReport, GeneratorError> {
    if config.workers == 0 {
        return Err(GeneratorError::InvalidInput { field: "workers" });
    }
    verify_hash(
        &config.v9_dataset_path,
        HISTORICAL_V9_DATASET_SHA256,
        "v9_dataset_sha256",
    )?;
    verify_hash(
        &config.v10_jar_path,
        BACKWARD_V10_JAR_SHA256,
        "backward_v10_jar_sha256",
    )?;
    verify_hash(
        &config.policy_packs_path,
        PINNED_POLICY_PACKS_SHA256,
        "policy_packs_sha256",
    )?;
    let divergent_control_jar_sha256 = file_sha256(&config.divergent_control_jar_path)?;
    if divergent_control_jar_sha256 == BACKWARD_V10_JAR_SHA256 {
        return Err(GeneratorError::InvalidInput {
            field: "divergent_control_jar_must_be_distinct",
        });
    }

    let dataset = PreparedDcgDataset::load(&config.v9_dataset_path).map_err(|error| {
        GeneratorError::Prepared {
            message: error.to_string(),
        }
    })?;
    let generation_seeds = dataset
        .records()
        .iter()
        .filter_map(|record| {
            record
                .generation
                .as_ref()
                .map(|value| value.generation_seed)
        })
        .collect::<BTreeSet<_>>();
    let [generation_seed] = generation_seeds.iter().copied().collect::<Vec<_>>()[..] else {
        return Err(GeneratorError::Prepared {
            message: "V9 replay requires exactly one generation seed".to_owned(),
        });
    };
    for record in dataset.records() {
        let generation = record
            .generation
            .as_ref()
            .ok_or_else(|| GeneratorError::Prepared {
                message: format!("record {} has no oracle provenance", record.record_id),
            })?;
        if generation.oracle_jar_sha256 != HISTORICAL_V9_JAR_SHA256
            || generation.policy_packs_sha256 != PINNED_POLICY_PACKS_SHA256
            || generation.compatibility_mode != "BACKWARD"
        {
            return Err(GeneratorError::Prepared {
                message: format!(
                    "record {} crosses the declared V9 identity boundary",
                    record.record_id
                ),
            });
        }
    }

    let source = JsonSchemaBenchSource {
        path: config.source_parquet_path.clone(),
        source: "jsonschemabench".to_owned(),
    };
    let seeds = source.load_sampled(usize::MAX, generation_seed)?;
    let source_parquet_families_loaded = seeds.len();
    let seed_map = seeds
        .into_iter()
        .map(|seed| (seed.id.clone(), seed))
        .collect::<BTreeMap<_, _>>();
    let mut candidate_cache = BTreeMap::<String, BTreeMap<(String, String), Arc<String>>>::new();
    let mut jobs = Vec::with_capacity(dataset.len());
    for (index, record) in dataset.records().iter().enumerate() {
        let generation = record
            .generation
            .as_ref()
            .expect("provenance checked above");
        let seed = seed_map
            .get(&record.family_id)
            .ok_or_else(|| GeneratorError::Prepared {
                message: format!("source parquet is missing V9 family {}", record.family_id),
            })?;
        let candidates = if let Some(candidates) = candidate_cache.get(&record.family_id) {
            candidates
        } else {
            let rebuilt = candidates_for_schema(&seed.schema)
                .map_err(|error| GeneratorError::Prepared {
                    message: format!("could not rebuild {}: {error}", record.family_id),
                })?
                .into_iter()
                .map(|candidate| {
                    (
                        (candidate.kind.as_str().to_owned(), candidate.variant),
                        Arc::new(candidate.candidate_schema),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            candidate_cache.insert(record.family_id.clone(), rebuilt);
            candidate_cache
                .get(&record.family_id)
                .expect("candidate cache was just inserted")
        };
        let key = (
            generation.declared_mutation.clone(),
            generation.mutation_variant.clone(),
        );
        let candidate_schema = candidates
            .get(&key)
            .ok_or_else(|| GeneratorError::Prepared {
                message: format!(
                    "source family {} no longer reproduces {}:{}",
                    record.family_id, generation.declared_mutation, generation.mutation_variant
                ),
            })?;
        let fingerprint = canonical_pair_fingerprint(
            &seed.schema,
            candidate_schema,
            &record.policy_pack,
            HISTORICAL_V9_JAR_SHA256,
        )?;
        if fingerprint != generation.pair_fingerprint {
            return Err(GeneratorError::Prepared {
                message: format!(
                    "historical pair fingerprint mismatch for {}",
                    record.record_id
                ),
            });
        }
        jobs.push(ReplayJob {
            index,
            record_id: record.record_id.clone(),
            family_id: record.family_id.clone(),
            policy_pack: record.policy_pack.clone(),
            declared_mutation: generation.declared_mutation.clone(),
            mutation_variant: generation.mutation_variant.clone(),
            pair_fingerprint: generation.pair_fingerprint.clone(),
            expected_outcome: generation.oracle_outcome.clone(),
            expected_stdout: generation.oracle_stdout.clone(),
            base_schema: Arc::new(seed.schema.clone()),
            candidate_schema: Arc::clone(candidate_schema),
        });
    }

    let oracle = PinnedOracle::new(OracleConfig {
        java_program: PathBuf::from("java"),
        jar_path: config.v10_jar_path.clone(),
        policy_packs_path: config.policy_packs_path.clone(),
    })?;
    let sensitivity_control = run_sensitivity_control(
        &config.workspace,
        &config.v10_jar_path,
        &config.divergent_control_jar_path,
        &divergent_control_jar_sha256,
    )?;
    let synthetic_negative_controls = run_synthetic_controls(&oracle, &config.workspace)?;
    if !sensitivity_control.divergence_detected
        || synthetic_negative_controls
            .iter()
            .any(|control| !control.passed)
    {
        return Err(GeneratorError::Prepared {
            message: "negative controls did not establish audit sensitivity".to_owned(),
        });
    }

    let invocations = run_replay_jobs(&oracle, &config.workspace, jobs, config.workers)?;
    let preserved_v9_replay = summarize(&invocations);
    let concrete_traces = build_traces(&dataset, &invocations);
    let all_outcomes_match = preserved_v9_replay.outcome_mismatches == 0;
    let all_stdout_match = preserved_v9_replay.exact_stdout_mismatches == 0;
    let controls_passed = sensitivity_control.divergence_detected
        && synthetic_negative_controls
            .iter()
            .all(|control| control.passed);
    let behavioral_result = if all_outcomes_match && all_stdout_match && controls_passed {
        "COMPLETE_PRESERVED_V9_MATCH"
    } else if all_outcomes_match && controls_passed {
        "DECISION_MATCH_OUTPUT_DIFFERENCE"
    } else {
        "BEHAVIORAL_DIVERGENCE"
    };
    Ok(BackwardEquivalenceAuditReport {
        format_version: "dcg-backward-v10-equivalence-audit-v1",
        status: "COMPLETE".to_owned(),
        identity_boundary: IdentityBoundary {
            historical_v9_jar_sha256: HISTORICAL_V9_JAR_SHA256,
            historical_v9_binary_available: false,
            backward_v10_jar_sha256: BACKWARD_V10_JAR_SHA256,
            permanent_rule: "The c00da951... binary is BACKWARD V10 only. It is never V9, never a replacement for 809b25e6...677dc, and never identity-interchangeable with frozen V9 evidence.",
        },
        inputs: AuditInputs {
            v9_dataset_path: config.v9_dataset_path.display().to_string(),
            v9_dataset_sha256: HISTORICAL_V9_DATASET_SHA256.to_owned(),
            source_parquet_path: config.source_parquet_path.display().to_string(),
            source_parquet_sha256: file_sha256(&config.source_parquet_path)?,
            v10_jar_path: config.v10_jar_path.display().to_string(),
            v10_jar_sha256: BACKWARD_V10_JAR_SHA256.to_owned(),
            divergent_control_jar_path: config.divergent_control_jar_path.display().to_string(),
            divergent_control_jar_sha256,
            policy_packs_path: config.policy_packs_path.display().to_string(),
            policy_packs_sha256: PINNED_POLICY_PACKS_SHA256.to_owned(),
            compatibility_mode: "BACKWARD",
            workers: config.workers,
        },
        reconstruction: ReconstructionSummary {
            records: dataset.len(),
            source_parquet_families_loaded,
            v9_families_reconstructed: dataset
                .records()
                .iter()
                .map(|record| record.family_id.as_str())
                .collect::<BTreeSet<_>>()
                .len(),
            generation_seed,
            historical_pair_fingerprints_verified: dataset.len(),
            historical_pair_fingerprint_failures: 0,
        },
        sensitivity_control,
        synthetic_negative_controls,
        preserved_v9_replay,
        invocations,
        concrete_traces,
        conclusion: AuditConclusion {
            behavioral_result: behavioral_result.to_owned(),
            all_preserved_v9_outcomes_match: all_outcomes_match,
            all_preserved_v9_stdout_match: all_stdout_match,
            negative_controls_passed: controls_passed,
            identity_interchangeable: false,
            allowed_name: "BACKWARD V10",
            prohibited_claim: "V9 restored or V10 interchangeable with V9",
        },
    })
}

fn run_replay_jobs(
    oracle: &PinnedOracle,
    workspace: &Path,
    jobs: Vec<ReplayJob>,
    workers: usize,
) -> Result<Vec<InvocationComparison>, GeneratorError> {
    let jobs = Arc::new(jobs);
    let next = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();
    let worker_count = workers.min(jobs.len()).max(1);
    thread::scope(|scope| {
        for _ in 0..worker_count {
            let jobs = Arc::clone(&jobs);
            let next = Arc::clone(&next);
            let sender = sender.clone();
            let oracle = oracle.clone();
            let workspace = workspace.to_path_buf();
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = jobs.get(index) else {
                        break;
                    };
                    let pair = GeneratedPair {
                        contract_id: format!("backward-v10-audit-{}", job.index),
                        policy_pack: job.policy_pack.clone(),
                        base_schema: job.base_schema.as_ref().clone(),
                        candidate_schema: job.candidate_schema.as_ref().clone(),
                    };
                    let result = oracle
                        .check(&workspace, &pair, OracleCompatibilityMode::Backward)
                        .map(|run| {
                            let actual_outcome = run.outcome.as_str().to_owned();
                            let outcome_match = actual_outcome == job.expected_outcome;
                            let stdout_match = run.stdout == job.expected_stdout;
                            let comparison = InvocationComparison {
                                index: job.index,
                                record_id: job.record_id.clone(),
                                family_id: job.family_id.clone(),
                                policy_pack: job.policy_pack.clone(),
                                declared_mutation: job.declared_mutation.clone(),
                                mutation_variant: job.mutation_variant.clone(),
                                historical_pair_fingerprint: job.pair_fingerprint.clone(),
                                expected_v9_outcome: job.expected_outcome.clone(),
                                actual_v10_outcome: actual_outcome,
                                exit_code: run.exit_code,
                                expected_stdout_sha256: sha256_text(&job.expected_stdout),
                                actual_stdout_sha256: sha256_text(&run.stdout),
                                outcome_match,
                                stdout_match,
                                stderr_empty: run.stderr.is_empty(),
                                rejection_stage: run.rejection_stage,
                                rejection_reason: run.rejection_reason,
                                mismatched_expected_stdout: (!stdout_match)
                                    .then(|| job.expected_stdout.clone()),
                                mismatched_actual_stdout: (!stdout_match)
                                    .then(|| run.stdout.clone()),
                            };
                            cleanup_staged_run(&run.staged_contract_dir);
                            comparison
                        });
                    if sender.send((job.index, result)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
        let mut completed = 0usize;
        let mut results = (0..jobs.len()).map(|_| None).collect::<Vec<_>>();
        for (index, result) in receiver {
            results[index] = Some(result?);
            completed += 1;
            if completed.is_multiple_of(1000) || completed == jobs.len() {
                println!(
                    "BACKWARD V10 equivalence replay progress: {completed}/{}",
                    jobs.len()
                );
            }
        }
        results
            .into_iter()
            .enumerate()
            .map(|(index, result)| {
                result.ok_or_else(|| GeneratorError::Prepared {
                    message: format!("worker did not return replay index {index}"),
                })
            })
            .collect::<Result<Vec<_>, _>>()
    })
}

fn run_sensitivity_control(
    workspace: &Path,
    v10_jar: &Path,
    divergent_jar: &Path,
    divergent_sha256: &str,
) -> Result<SensitivityControl, GeneratorError> {
    let probe_dir = workspace.join("policy-fallback-sensitivity-probe");
    fs::create_dir_all(&probe_dir).map_err(io_error)?;
    let source = probe_dir.join("PolicyFallbackProbe.java");
    fs::write(
        &source,
        r#"import com.ideas.contracts.core.PolicyPack;
import com.ideas.contracts.core.RuleId;
import java.util.Map;

public class PolicyFallbackProbe {
  public static void main(String[] args) {
    System.out.println(new PolicyPack("partial", Map.of()).severityFor(RuleId.ENUM_VALUE_ADDED));
  }
}
"#,
    )
    .map_err(io_error)?;
    let compile = Command::new("javac")
        .arg("-cp")
        .arg(v10_jar)
        .arg(&source)
        .output()
        .map_err(spawn_error)?;
    if !compile.status.success() {
        return Err(GeneratorError::Prepared {
            message: format!(
                "could not compile sensitivity probe: {}",
                String::from_utf8_lossy(&compile.stderr)
            ),
        });
    }
    let v10_result = run_probe(&probe_dir, v10_jar)?;
    let divergent_result = run_probe(&probe_dir, divergent_jar)?;
    Ok(SensitivityControl {
        id: "partial-policy-omitted-enum-added",
        shared_code_path: "PolicyPack.severityFor(RuleId.ENUM_VALUE_ADDED)",
        mechanism: "The same compiled probe constructs a deliberately partial PolicyPack and is executed once against each JAR. The comparator must observe BREAKING versus WARNING.",
        divergence_detected: v10_result == "BREAKING"
            && divergent_result == "WARNING"
            && v10_result != divergent_result,
        v10_result,
        known_divergent_jar_sha256: divergent_sha256.to_owned(),
        known_divergent_result: divergent_result,
        scope_note: "PolicyPackConfig fills omitted CLI policy entries from baseline defaults, so this shared-code divergence is a programmatic-API risk rather than a difference expected in the 15-pack CLI replay. The control proves the audit exercises the code that actually changed.",
    })
}

fn run_probe(probe_dir: &Path, jar: &Path) -> Result<String, GeneratorError> {
    let separator = if cfg!(windows) { ";" } else { ":" };
    let classpath = format!("{}{}{}", probe_dir.display(), separator, jar.display());
    let output = Command::new("java")
        .arg("-cp")
        .arg(classpath)
        .arg("PolicyFallbackProbe")
        .output()
        .map_err(spawn_error)?;
    if !output.status.success() {
        return Err(GeneratorError::Prepared {
            message: format!(
                "sensitivity probe failed against {}: {}",
                jar.display(),
                String::from_utf8_lossy(&output.stderr)
            ),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn run_synthetic_controls(
    oracle: &PinnedOracle,
    workspace: &Path,
) -> Result<Vec<SyntheticControl>, GeneratorError> {
    let cases = synthetic_cases();
    let mut controls = Vec::with_capacity(cases.len());
    for case in cases {
        let pair = GeneratedPair {
            contract_id: format!("backward-v10-control-{}", case.id),
            policy_pack: "baseline".to_owned(),
            base_schema: case.base.to_owned(),
            candidate_schema: case.candidate.to_owned(),
        };
        let run = oracle.check(workspace, &pair, OracleCompatibilityMode::Backward)?;
        let actual_outcome = run.outcome.as_str().to_owned();
        let fragment_matches = case
            .expected_message_fragment
            .is_none_or(|fragment| run.stdout.contains(fragment));
        let passed = actual_outcome == case.expected_outcome
            && fragment_matches
            && run.outcome != OracleOutcome::Rejected;
        controls.push(SyntheticControl {
            id: case.id.to_owned(),
            purpose: case.purpose.to_owned(),
            expected_outcome: case.expected_outcome.to_owned(),
            expected_message_fragment: case.expected_message_fragment.map(str::to_owned),
            actual_outcome,
            exit_code: run.exit_code,
            stdout: run.stdout,
            stderr: run.stderr,
            rejection_stage: run.rejection_stage,
            rejection_reason: run.rejection_reason,
            passed,
            base_schema_sha256: sha256_text(case.base),
            candidate_schema_sha256: sha256_text(case.candidate),
        });
        cleanup_staged_run(&run.staged_contract_dir);
    }
    Ok(controls)
}

fn synthetic_cases() -> Vec<SyntheticCase> {
    vec![
        SyntheticCase {
            id: "root-optional-addition-open",
            purpose: "Detect FORWARD optional-addition logic leaking into BACKWARD for an open object.",
            base: r#"{"type":"object","additionalProperties":true,"properties":{"id":{"type":"string"}}}"#,
            candidate: r#"{"type":"object","additionalProperties":true,"properties":{"id":{"type":"string"},"note":{"type":"string"}}}"#,
            expected_outcome: "safe",
            expected_message_fragment: Some("Schema compatibility: PASS"),
        },
        SyntheticCase {
            id: "root-optional-addition-closed",
            purpose: "Detect FORWARD optional-addition logic leaking into BACKWARD for a closed object.",
            base: r#"{"type":"object","additionalProperties":false,"properties":{"id":{"type":"string"}}}"#,
            candidate: r#"{"type":"object","additionalProperties":false,"properties":{"id":{"type":"string"},"note":{"type":"string"}}}"#,
            expected_outcome: "safe",
            expected_message_fragment: Some("Schema compatibility: PASS"),
        },
        SyntheticCase {
            id: "root-optional-addition-omitted-profile",
            purpose: "Exercise the omitted additionalProperties BACKWARD path.",
            base: r#"{"type":"object","properties":{"id":{"type":"string"}}}"#,
            candidate: r#"{"type":"object","properties":{"id":{"type":"string"},"note":{"type":"string"}}}"#,
            expected_outcome: "safe",
            expected_message_fragment: Some("Schema compatibility: PASS"),
        },
        SyntheticCase {
            id: "nested-optional-addition-closed",
            purpose: "Exercise nested diff construction while detecting FORWARD-rule leakage.",
            base: r#"{"type":"object","properties":{"profile":{"type":"object","additionalProperties":false,"properties":{"id":{"type":"string"}}}}}"#,
            candidate: r#"{"type":"object","properties":{"profile":{"type":"object","additionalProperties":false,"properties":{"id":{"type":"string"},"note":{"type":"string"}}}}}"#,
            expected_outcome: "safe",
            expected_message_fragment: Some("Schema compatibility: PASS"),
        },
        SyntheticCase {
            id: "root-optional-field-removal",
            purpose: "Prove genuine BACKWARD removal attribution remains FIELD_REMOVED.",
            base: r#"{"type":"object","properties":{"id":{"type":"string"},"note":{"type":"string"}}}"#,
            candidate: r#"{"type":"object","properties":{"id":{"type":"string"}}}"#,
            expected_outcome: "breaking",
            expected_message_fragment: Some("Field removed: note"),
        },
        SyntheticCase {
            id: "nested-required-field-removal",
            purpose: "Exercise nested required-property removal and exact FIELD_REMOVED attribution.",
            base: r#"{"type":"object","properties":{"profile":{"type":"object","properties":{"id":{"type":"string"},"name":{"type":"string"}},"required":["id","name"]}}}"#,
            candidate: r#"{"type":"object","properties":{"profile":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}}}"#,
            expected_outcome: "breaking",
            expected_message_fragment: Some("Field removed: profile.name"),
        },
        SyntheticCase {
            id: "genuine-required-field-addition",
            purpose: "Prove a genuine required addition remains REQUIRED_FIELD_ADDED in BACKWARD.",
            base: r#"{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}"#,
            candidate: r#"{"type":"object","properties":{"id":{"type":"string"},"region":{"type":"string"}},"required":["id","region"]}"#,
            expected_outcome: "breaking",
            expected_message_fragment: Some("Required field added: region"),
        },
        SyntheticCase {
            id: "nested-field-type-change",
            purpose: "Exercise shared nested-object diff construction.",
            base: r#"{"type":"object","properties":{"profile":{"type":"object","properties":{"age":{"type":"integer"}}}}}"#,
            candidate: r#"{"type":"object","properties":{"profile":{"type":"object","properties":{"age":{"type":"string"}}}}}"#,
            expected_outcome: "breaking",
            expected_message_fragment: Some("Field type changed: profile.age"),
        },
        SyntheticCase {
            id: "array-item-field-type-change",
            purpose: "Exercise shared array-item diff paths.",
            base: r#"{"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"quantity":{"type":"integer"}}}}}}"#,
            candidate: r#"{"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"quantity":{"type":"string"}}}}}}"#,
            expected_outcome: "breaking",
            expected_message_fragment: Some("Field type changed: items[].quantity"),
        },
        SyntheticCase {
            id: "nested-constraint-tightening",
            purpose: "Exercise shared nested constraint diff construction.",
            base: r#"{"type":"object","properties":{"profile":{"type":"object","properties":{"name":{"type":"string","maxLength":20}}}}}"#,
            candidate: r#"{"type":"object","properties":{"profile":{"type":"object","properties":{"name":{"type":"string","maxLength":10}}}}}"#,
            expected_outcome: "breaking",
            expected_message_fragment: Some(
                "Constraint tightened: profile.name.maxLength (20 -> 10)",
            ),
        },
    ]
}

fn summarize(invocations: &[InvocationComparison]) -> ReplaySummary {
    let mut summary = ReplaySummary {
        invocations: invocations.len(),
        accepted: 0,
        rejected: 0,
        outcome_matches: 0,
        outcome_mismatches: 0,
        exact_stdout_matches: 0,
        exact_stdout_mismatches: 0,
        stderr_nonempty: 0,
        by_mutation: BTreeMap::new(),
        by_policy_pack: BTreeMap::new(),
    };
    for invocation in invocations {
        let rejected = invocation.actual_v10_outcome == "rejected";
        summary.accepted += usize::from(!rejected);
        summary.rejected += usize::from(rejected);
        summary.outcome_matches += usize::from(invocation.outcome_match);
        summary.outcome_mismatches += usize::from(!invocation.outcome_match);
        summary.exact_stdout_matches += usize::from(invocation.stdout_match);
        summary.exact_stdout_mismatches += usize::from(!invocation.stdout_match);
        summary.stderr_nonempty += usize::from(!invocation.stderr_empty);
        update_counts(
            summary
                .by_mutation
                .entry(invocation.declared_mutation.clone())
                .or_default(),
            invocation,
        );
        update_counts(
            summary
                .by_policy_pack
                .entry(invocation.policy_pack.clone())
                .or_default(),
            invocation,
        );
    }
    summary
}

fn update_counts(counts: &mut OutcomeCounts, invocation: &InvocationComparison) {
    counts.invocations += 1;
    match invocation.actual_v10_outcome.as_str() {
        "safe" => counts.safe += 1,
        "warning" => counts.warning += 1,
        "breaking" => counts.breaking += 1,
        _ => counts.rejected += 1,
    }
    counts.outcome_mismatches += usize::from(!invocation.outcome_match);
    counts.stdout_mismatches += usize::from(!invocation.stdout_match);
}

fn build_traces(
    dataset: &PreparedDcgDataset,
    invocations: &[InvocationComparison],
) -> Vec<ConcreteTrace> {
    let mut seen = BTreeSet::new();
    let mut traces = Vec::new();
    for invocation in invocations {
        if !seen.insert(invocation.declared_mutation.clone()) {
            continue;
        }
        let record = &dataset.records()[invocation.index];
        let generation = record
            .generation
            .as_ref()
            .expect("V9 provenance was checked");
        traces.push(ConcreteTrace {
            record_id: invocation.record_id.clone(),
            family_id: invocation.family_id.clone(),
            policy_pack: invocation.policy_pack.clone(),
            declared_mutation: invocation.declared_mutation.clone(),
            mutation_variant: invocation.mutation_variant.clone(),
            historical_pair_fingerprint: invocation.historical_pair_fingerprint.clone(),
            expected_v9_outcome: invocation.expected_v9_outcome.clone(),
            actual_v10_outcome: invocation.actual_v10_outcome.clone(),
            expected_v9_stdout: generation.oracle_stdout.clone(),
            actual_v10_stdout: invocation
                .mismatched_actual_stdout
                .clone()
                .unwrap_or_else(|| generation.oracle_stdout.clone()),
        });
    }
    traces
}

fn cleanup_staged_run(contract_dir: &Path) {
    if let Some(root) = contract_dir.parent().and_then(Path::parent) {
        let _ = fs::remove_dir_all(root);
    }
}

fn verify_hash(path: &Path, expected: &str, field: &'static str) -> Result<(), GeneratorError> {
    if file_sha256(path)? != expected {
        return Err(GeneratorError::InvalidInput { field });
    }
    Ok(())
}

fn file_sha256(path: &Path) -> Result<String, GeneratorError> {
    let bytes = fs::read(path).map_err(io_error)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn sha256_text(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}

fn io_error(error: std::io::Error) -> GeneratorError {
    GeneratorError::Io {
        message: error.to_string(),
    }
}

fn spawn_error(error: std::io::Error) -> GeneratorError {
    GeneratorError::Spawn {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_control_matrix_covers_every_declared_shared_risk() {
        let ids = synthetic_cases()
            .into_iter()
            .map(|case| case.id)
            .collect::<BTreeSet<_>>();
        for required in [
            "root-optional-addition-open",
            "root-optional-addition-closed",
            "root-optional-addition-omitted-profile",
            "nested-optional-addition-closed",
            "root-optional-field-removal",
            "nested-required-field-removal",
            "genuine-required-field-addition",
            "nested-field-type-change",
            "array-item-field-type-change",
            "nested-constraint-tightening",
        ] {
            assert!(
                ids.contains(required),
                "missing negative control {required}"
            );
        }
    }

    #[test]
    fn frozen_identities_are_never_equal() {
        assert_ne!(HISTORICAL_V9_JAR_SHA256, BACKWARD_V10_JAR_SHA256);
    }
}
