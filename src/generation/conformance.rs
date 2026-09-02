//! Full-mode, full-policy executable-oracle conformance matrix generation.

use super::{GeneratedPair, GeneratorError, PinnedOracle, SeedSchema, candidates_for_schema};
use crate::models::{
    InvariantEvidenceConfig, OracleCompatibilityMode, OracleConformancePreflight,
    OracleConformanceReport, OracleConformanceRun,
};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

/// Conservative cap for independent external-JAR invocations. The final
/// report is sorted by stable evidence identity, so completion order cannot
/// affect its portable content.
const MAX_CONFORMANCE_WORKERS: usize = 8;

#[derive(Debug, Clone)]
struct ConformanceJob {
    family_id: String,
    mutation_family: String,
    mutation_variant: String,
    consumer_profile: String,
    base_schema: String,
    candidate_schema: String,
}

/// Runs every applicable single mutation through every requested policy and
/// compatibility direction. The returned artifact retains raw executable
/// evidence, including rejected calls, rather than inferring any result.
pub fn build_oracle_conformance_matrix(
    oracle: &PinnedOracle,
    workspace: &Path,
    policies: &[String],
    modes: &[OracleCompatibilityMode],
    mutation_families: &[String],
    thresholds: InvariantEvidenceConfig,
    seeds: impl IntoIterator<Item = SeedSchema>,
) -> Result<OracleConformanceReport, GeneratorError> {
    if policies.is_empty() || policies.iter().any(|policy| policy.trim().is_empty()) {
        return Err(GeneratorError::InvalidInput {
            field: "policy_packs",
        });
    }
    if modes.is_empty() {
        return Err(GeneratorError::InvalidInput {
            field: "compatibility_modes",
        });
    }
    let mut jobs = Vec::new();
    for seed in seeds {
        if seed.id.trim().is_empty() {
            continue;
        }
        let Ok(candidates) = candidates_for_schema(&seed.schema) else {
            continue;
        };
        for candidate in candidates {
            if !mutation_families.is_empty()
                && !mutation_families
                    .iter()
                    .any(|family| family == candidate.kind.as_str())
            {
                continue;
            }
            jobs.push(ConformanceJob {
                family_id: seed.id.clone(),
                mutation_family: candidate.kind.as_str().to_owned(),
                mutation_variant: candidate.variant,
                consumer_profile: consumer_profile(&seed.schema).to_owned(),
                base_schema: seed.schema.clone(),
                candidate_schema: candidate.candidate_schema,
            });
        }
    }
    let (preflight_runs, jobs) = preflight_jobs(oracle, workspace, &policies[0], jobs, false)?;
    let runs = run_jobs_in_parallel(oracle, workspace, policies, modes, jobs)?;
    Ok(OracleConformanceReport::new_with_preflights(
        oracle.jar_sha256().to_owned(),
        oracle.policy_packs_sha256().to_owned(),
        thresholds,
        preflight_runs,
        runs,
    ))
}

/// Derives explicit open and closed consumers from every source family, then
/// admits a family only when every generated profile/variant pair passes the
/// separate schema preflight. This preserves matched-family comparisons.
pub fn build_matched_optional_field_conformance_matrix(
    oracle: &PinnedOracle,
    workspace: &Path,
    policies: &[String],
    modes: &[OracleCompatibilityMode],
    thresholds: InvariantEvidenceConfig,
    max_families: usize,
    seeds: impl IntoIterator<Item = SeedSchema>,
) -> Result<OracleConformanceReport, GeneratorError> {
    if policies.is_empty() || policies.iter().any(|policy| policy.trim().is_empty()) {
        return Err(GeneratorError::InvalidInput {
            field: "policy_packs",
        });
    }
    if modes.is_empty() {
        return Err(GeneratorError::InvalidInput {
            field: "compatibility_modes",
        });
    }
    if max_families == 0 {
        return Err(GeneratorError::InvalidInput {
            field: "max_families",
        });
    }
    let mut jobs = Vec::new();
    for seed in seeds {
        for (consumer_profile, closed) in [("open", false), ("closed", true)] {
            let Some(base_schema) = with_root_profile(&seed.schema, closed) else {
                continue;
            };
            let Ok(candidates) = candidates_for_schema(&base_schema) else {
                continue;
            };
            for candidate in candidates
                .into_iter()
                .filter(|candidate| candidate.kind.as_str() == "optional_field_added")
            {
                jobs.push(ConformanceJob {
                    family_id: seed.id.clone(),
                    mutation_family: candidate.kind.as_str().to_owned(),
                    mutation_variant: candidate.variant,
                    consumer_profile: consumer_profile.to_owned(),
                    base_schema: base_schema.clone(),
                    candidate_schema: candidate.candidate_schema,
                });
            }
        }
    }
    let (preflight_runs, mut jobs) = preflight_jobs(oracle, workspace, &policies[0], jobs, true)?;
    retain_first_families(&mut jobs, max_families);
    let runs = run_jobs_in_parallel(oracle, workspace, policies, modes, jobs)?;
    Ok(OracleConformanceReport::new_with_preflights(
        oracle.jar_sha256().to_owned(),
        oracle.policy_packs_sha256().to_owned(),
        thresholds,
        preflight_runs,
        runs,
    ))
}

fn retain_first_families(jobs: &mut Vec<ConformanceJob>, max_families: usize) {
    let mut selected = Vec::new();
    for job in jobs.iter() {
        if !selected.contains(&job.family_id) {
            selected.push(job.family_id.clone());
            if selected.len() == max_families {
                break;
            }
        }
    }
    jobs.retain(|job| selected.contains(&job.family_id));
}

fn preflight_jobs(
    oracle: &PinnedOracle,
    workspace: &Path,
    policy_pack: &str,
    jobs: Vec<ConformanceJob>,
    require_complete_families: bool,
) -> Result<(Vec<OracleConformancePreflight>, Vec<ConformanceJob>), GeneratorError> {
    let mut accepted = Vec::new();
    let mut preflights = Vec::new();
    let mut rejected_families = BTreeSet::new();
    for job in jobs {
        let run = oracle.preflight(
            workspace,
            &GeneratedPair {
                contract_id: job.family_id.clone(),
                policy_pack: policy_pack.to_owned(),
                base_schema: job.base_schema.clone(),
                candidate_schema: job.candidate_schema.clone(),
            },
        )?;
        if run.accepted {
            accepted.push(job.clone());
        } else {
            rejected_families.insert(job.family_id.clone());
        }
        preflights.push(OracleConformancePreflight {
            family_id: job.family_id,
            mutation_family: job.mutation_family,
            mutation_variant: job.mutation_variant,
            consumer_profile: job.consumer_profile,
            accepted: run.accepted,
            exit_code: run.exit_code,
            stdout: run.stdout,
            stderr: run.stderr,
            rejection_stage: run.rejection_stage,
            rejection_reason: run.rejection_reason,
        });
    }
    if require_complete_families {
        accepted.retain(|job| !rejected_families.contains(&job.family_id));
    }
    Ok((preflights, accepted))
}

fn with_root_profile(schema: &str, closed: bool) -> Option<String> {
    let mut value = serde_json::from_str::<Value>(schema).ok()?;
    let root = value.as_object_mut()?;
    if root.get("properties")?.as_object()?.is_empty() {
        return None;
    }
    root.insert("additionalProperties".to_owned(), Value::Bool(!closed));
    serde_json::to_string(&value).ok()
}

fn consumer_profile(schema: &str) -> &'static str {
    let Ok(Value::Object(root)) = serde_json::from_str::<Value>(schema) else {
        return "unknown";
    };
    if root.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
        "closed"
    } else {
        "open"
    }
}

#[derive(Clone, Copy)]
struct RemovalAuditCase {
    family_id: &'static str,
    case_id: &'static str,
    mutation_kind: &'static str,
    structural_scope: &'static str,
    expected_path: &'static str,
    base_schema: &'static str,
    candidate_schema: &'static str,
}

/// Runs the small executable-JAR audit that isolates genuine removals from a
/// genuine required-field addition. Production-policy calls and mechanism-only
/// severity-fixture calls remain distinct in every persisted record.
pub fn build_forward_removal_mechanism_audit(
    pinned_oracle: &PinnedOracle,
    audit_oracle: &PinnedOracle,
    workspace: &Path,
) -> Result<Value, GeneratorError> {
    if pinned_oracle.jar_sha256() != audit_oracle.jar_sha256() {
        return Err(GeneratorError::InvalidInput {
            field: "audit_oracles_must_use_the_same_jar",
        });
    }
    let cases = removal_audit_cases();
    let mut preflights = Vec::new();
    let mut invocations = Vec::new();
    let mut assertions = Vec::new();

    for case in &cases {
        let mut policies = vec![(pinned_oracle, "pinned-production", "baseline", "BREAKING")];
        if case.mutation_kind == "FIELD_REMOVED" {
            policies.extend([
                (
                    audit_oracle,
                    "audit-mechanism-only",
                    "field-removed-warning-audit",
                    "WARNING",
                ),
                (
                    audit_oracle,
                    "audit-mechanism-only",
                    "field-removed-ignore-audit",
                    "IGNORE",
                ),
            ]);
        }
        for (oracle, policy_source, policy_pack, configured_severity) in policies {
            let pair = GeneratedPair {
                contract_id: case.family_id.to_owned(),
                policy_pack: policy_pack.to_owned(),
                base_schema: case.base_schema.to_owned(),
                candidate_schema: case.candidate_schema.to_owned(),
            };
            let preflight = oracle.preflight(&workspace.join(policy_source), &pair)?;
            preflights.push(serde_json::json!({
                "family_id": case.family_id,
                "case_id": case.case_id,
                "mutation_kind": case.mutation_kind,
                "structural_scope": case.structural_scope,
                "policy_source": policy_source,
                "policy_pack": policy_pack,
                "configured_severity": configured_severity,
                "policy_packs_sha256": oracle.policy_packs_sha256(),
                "accepted": preflight.accepted,
                "exit_code": preflight.exit_code,
                "stdout": preflight.stdout,
                "stderr": preflight.stderr,
                "rejection_stage": preflight.rejection_stage,
                "rejection_reason": preflight.rejection_reason,
            }));
            for mode in [
                OracleCompatibilityMode::Forward,
                OracleCompatibilityMode::Full,
            ] {
                let run = oracle.check(&workspace.join(policy_source), &pair, mode)?;
                let outcome = run.outcome.as_str();
                let expected_phrase = if case.mutation_kind == "FIELD_REMOVED" {
                    format!("Field removed: {}", case.expected_path)
                } else {
                    format!("Required field added: {}", case.expected_path)
                };
                let expected_outcome = match configured_severity {
                    "BREAKING" => "breaking",
                    "WARNING" => "warning",
                    "IGNORE" => "safe",
                    _ => unreachable!("audit severity list is fixed"),
                };
                let expected_finding_count = if configured_severity == "IGNORE" {
                    0
                } else if mode == OracleCompatibilityMode::Full {
                    2
                } else {
                    1
                };
                let finding_count = run.stdout.matches(&expected_phrase).count();
                let wrong_rule_absent = if case.mutation_kind == "FIELD_REMOVED" {
                    !run.stdout.contains("Required field added:")
                } else {
                    !run.stdout.contains("Field removed:")
                };
                let forward_attribution_present = configured_severity == "IGNORE"
                    || run.stdout.contains(&format!("[FORWARD] {expected_phrase}"));
                let backward_attribution_for_full = configured_severity == "IGNORE"
                    || mode != OracleCompatibilityMode::Full
                    || run.stdout.contains(&expected_phrase);
                let passed = run.rejection_stage.is_none()
                    && outcome == expected_outcome
                    && finding_count == expected_finding_count
                    && wrong_rule_absent
                    && forward_attribution_present
                    && backward_attribution_for_full;
                assertions.push(serde_json::json!({
                    "case_id": case.case_id,
                    "compatibility_mode": mode,
                    "policy_source": policy_source,
                    "policy_pack": policy_pack,
                    "configured_severity": configured_severity,
                    "expected_rule_id": case.mutation_kind,
                    "expected_outcome": expected_outcome,
                    "actual_outcome": outcome,
                    "expected_finding_count": expected_finding_count,
                    "actual_finding_count": finding_count,
                    "wrong_rule_absent": wrong_rule_absent,
                    "forward_attribution_present": forward_attribution_present,
                    "full_backward_attribution_present": backward_attribution_for_full,
                    "passed": passed,
                }));
                invocations.push(serde_json::json!({
                    "family_id": case.family_id,
                    "case_id": case.case_id,
                    "mutation_kind": case.mutation_kind,
                    "structural_scope": case.structural_scope,
                    "expected_path": case.expected_path,
                    "compatibility_mode": mode,
                    "policy_source": policy_source,
                    "policy_pack": policy_pack,
                    "configured_severity": configured_severity,
                    "oracle_jar_sha256": oracle.jar_sha256(),
                    "policy_packs_sha256": oracle.policy_packs_sha256(),
                    "exit_code": run.exit_code,
                    "outcome": outcome,
                    "stdout": run.stdout,
                    "stderr": run.stderr,
                    "rejection_stage": run.rejection_stage,
                    "rejection_reason": run.rejection_reason,
                }));
            }
        }
    }

    let preflight_passed = preflights
        .iter()
        .all(|record| record["accepted"].as_bool() == Some(true));
    let assertions_passed = assertions
        .iter()
        .all(|record| record["passed"].as_bool() == Some(true));
    let pinned_policy = serde_json::from_slice::<Value>(
        &std::fs::read(pinned_oracle.policy_packs_path()).map_err(|error| GeneratorError::Io {
            message: error.to_string(),
        })?,
    )
    .map_err(|error| GeneratorError::Io {
        message: error.to_string(),
    })?;
    let pinned_pack_count = pinned_policy["packs"]
        .as_object()
        .map_or(0, |packs| packs.len());
    let non_breaking_field_removed_overrides = pinned_policy["packs"]
        .as_object()
        .into_iter()
        .flat_map(|packs| packs.values())
        .filter(|pack| {
            pack["rules"]["FIELD_REMOVED"]
                .as_str()
                .is_some_and(|severity| severity != "BREAKING")
        })
        .count();

    Ok(serde_json::json!({
        "format_version": "forward-removal-mechanism-audit-v1",
        "purpose": "Executable-JAR verification of FORWARD/FULL removal dispatch, required-addition attribution, and FIELD_REMOVED severity resolution.",
        "production_policy_claim": false,
        "passed": preflight_passed && assertions_passed,
        "identity": {
            "oracle_jar_sha256": pinned_oracle.jar_sha256(),
            "pinned_policy_packs_sha256": pinned_oracle.policy_packs_sha256(),
            "audit_only_policy_fixture_sha256": audit_oracle.policy_packs_sha256(),
        },
        "policy_boundary": {
            "pinned_production_calls": "baseline BREAKING calls only",
            "audit_mechanism_only_calls": "WARNING and IGNORE resolver checks only",
            "corpus_or_benchmark_use_of_audit_fixture_allowed": false,
            "enforcement": "PinnedOracle::new rejects the audit fixture by SHA-256 even if renamed; only new_policy_mechanism_audit accepts its exact file name and hash."
        },
        "pinned_policy_gap": {
            "packs": pinned_pack_count,
            "field_removed_baseline_severity": pinned_policy["packs"]["baseline"]["rules"]["FIELD_REMOVED"],
            "non_breaking_field_removed_overrides": non_breaking_field_removed_overrides,
            "finding": "The pinned file resolves FIELD_REMOVED to BREAKING across all packs and defines no WARNING or IGNORE variant.",
            "governance_intent": "UNRESOLVED: repository documentation does not say whether always-BREAKING removal is deliberate governance or an omission; owner confirmation is required."
        },
        "full_mode_duplication": {
            "expected": true,
            "decision": "BY_DESIGN",
            "rationale": "FULL preserves independently attributed BACKWARD and [FORWARD] findings for a change that is evaluated in both directions; string-level deduplication would erase direction evidence."
        },
        "case_count": cases.len(),
        "independent_families": cases.iter().map(|case| case.family_id).collect::<BTreeSet<_>>().len(),
        "preflights": preflights,
        "invocations": invocations,
        "assertions": assertions,
    }))
}

fn removal_audit_cases() -> Vec<RemovalAuditCase> {
    vec![
        RemovalAuditCase {
            family_id: "audit.root.optional",
            case_id: "root-optional-removal",
            mutation_kind: "FIELD_REMOVED",
            structural_scope: "root_optional",
            expected_path: "removed",
            base_schema: r#"{"type":"object","properties":{"id":{"type":"string"},"removed":{"type":"string"}}}"#,
            candidate_schema: r#"{"type":"object","properties":{"id":{"type":"string"}}}"#,
        },
        RemovalAuditCase {
            family_id: "audit.root.required",
            case_id: "root-required-removal",
            mutation_kind: "FIELD_REMOVED",
            structural_scope: "root_required",
            expected_path: "removed",
            base_schema: r#"{"type":"object","properties":{"id":{"type":"string"},"removed":{"type":"string"}},"required":["removed"]}"#,
            candidate_schema: r#"{"type":"object","properties":{"id":{"type":"string"}}}"#,
        },
        RemovalAuditCase {
            family_id: "audit.nested.optional",
            case_id: "nested-optional-removal",
            mutation_kind: "FIELD_REMOVED",
            structural_scope: "nested_optional",
            expected_path: "outer.removed",
            base_schema: r#"{"type":"object","properties":{"outer":{"type":"object","properties":{"kept":{"type":"string"},"removed":{"type":"string"}}}}}"#,
            candidate_schema: r#"{"type":"object","properties":{"outer":{"type":"object","properties":{"kept":{"type":"string"}}}}}"#,
        },
        RemovalAuditCase {
            family_id: "audit.nested.required",
            case_id: "nested-required-removal",
            mutation_kind: "FIELD_REMOVED",
            structural_scope: "nested_required",
            expected_path: "outer.removed",
            base_schema: r#"{"type":"object","properties":{"outer":{"type":"object","properties":{"kept":{"type":"string"},"removed":{"type":"string"}},"required":["removed"]}},"required":["outer"]}"#,
            candidate_schema: r#"{"type":"object","properties":{"outer":{"type":"object","properties":{"kept":{"type":"string"}}}},"required":["outer"]}"#,
        },
        RemovalAuditCase {
            family_id: "audit.root.requiredaddition",
            case_id: "genuine-required-field-addition",
            mutation_kind: "REQUIRED_FIELD_ADDED",
            structural_scope: "root_required_addition",
            expected_path: "added",
            base_schema: r#"{"type":"object","properties":{"id":{"type":"string"}}}"#,
            candidate_schema: r#"{"type":"object","properties":{"id":{"type":"string"},"added":{"type":"string"}},"required":["added"]}"#,
        },
    ]
}

fn run_jobs_in_parallel(
    oracle: &PinnedOracle,
    workspace: &Path,
    policies: &[String],
    modes: &[OracleCompatibilityMode],
    jobs: Vec<ConformanceJob>,
) -> Result<Vec<OracleConformanceRun>, GeneratorError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    let worker_count = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(MAX_CONFORMANCE_WORKERS)
        .min(jobs.len());
    let jobs = Arc::new(jobs);
    let next_job = AtomicUsize::new(0);
    let runs = Mutex::new(Vec::new());
    let failure = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            let worker_oracle = oracle.clone();
            let worker_workspace = workspace.to_path_buf();
            let worker_jobs = Arc::clone(&jobs);
            let worker_next_job = &next_job;
            let worker_runs = &runs;
            let worker_failure = &failure;
            scope.spawn(move || {
                loop {
                    if worker_failure
                        .lock()
                        .expect("failure mutex poisoned")
                        .is_some()
                    {
                        return;
                    }
                    let index = worker_next_job.fetch_add(1, Ordering::Relaxed);
                    let Some(job) = worker_jobs.get(index) else {
                        return;
                    };
                    let mut local_runs = Vec::new();
                    for &mode in modes {
                        for policy_pack in policies {
                            match worker_oracle.check(
                                &worker_workspace,
                                &GeneratedPair {
                                    contract_id: job.family_id.clone(),
                                    policy_pack: policy_pack.clone(),
                                    base_schema: job.base_schema.clone(),
                                    candidate_schema: job.candidate_schema.clone(),
                                },
                                mode,
                            ) {
                                Ok(run) => local_runs.push(OracleConformanceRun {
                                    family_id: job.family_id.clone(),
                                    mutation_family: job.mutation_family.clone(),
                                    mutation_variant: job.mutation_variant.clone(),
                                    consumer_profile: job.consumer_profile.clone(),
                                    compatibility_mode: mode,
                                    policy_pack: policy_pack.clone(),
                                    exit_code: run.exit_code,
                                    outcome: run.outcome.as_str().to_owned(),
                                    stdout: run.stdout,
                                    stderr: run.stderr,
                                    rejection_stage: run.rejection_stage,
                                    rejection_reason: run.rejection_reason,
                                }),
                                Err(error) => {
                                    let mut stored =
                                        worker_failure.lock().expect("failure mutex poisoned");
                                    if stored.is_none() {
                                        *stored = Some(error);
                                    }
                                    return;
                                }
                            }
                        }
                    }
                    worker_runs
                        .lock()
                        .expect("runs mutex poisoned")
                        .extend(local_runs);
                }
            });
        }
    });
    if let Some(error) = failure.into_inner().expect("failure mutex poisoned") {
        return Err(error);
    }
    Ok(runs.into_inner().expect("runs mutex poisoned"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_profiles_keep_the_same_schema_family_and_optional_variants() {
        let source = r#"{
          "type": "object",
          "additionalProperties": {"type": "string"},
          "properties": {"id": {"type": "string"}}
        }"#;
        let open = with_root_profile(source, false).expect("open profile");
        let closed = with_root_profile(source, true).expect("closed profile");

        assert_eq!(consumer_profile(&open), "open");
        assert_eq!(consumer_profile(&closed), "closed");
        let variants = |schema: &str| {
            candidates_for_schema(schema)
                .expect("valid derived schema")
                .into_iter()
                .filter(|candidate| candidate.kind.as_str() == "optional_field_added")
                .map(|candidate| candidate.variant)
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(variants(&open), variants(&closed));
        assert_eq!(variants(&open).len(), 4);
    }

    #[test]
    fn final_family_sampling_happens_after_preflight_admission() {
        let job = |family: &str, variant: &str| ConformanceJob {
            family_id: family.to_owned(),
            mutation_family: "optional_field_added".to_owned(),
            mutation_variant: variant.to_owned(),
            consumer_profile: "open".to_owned(),
            base_schema: "{}".to_owned(),
            candidate_schema: "{}".to_owned(),
        };
        let mut jobs = vec![
            job("accepted-first", "string"),
            job("accepted-first", "array"),
            job("accepted-backfill", "string"),
            job("not-selected", "string"),
        ];

        retain_first_families(&mut jobs, 2);

        assert_eq!(
            jobs.iter()
                .map(|job| job.family_id.as_str())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["accepted-backfill", "accepted-first"])
        );
    }
}
