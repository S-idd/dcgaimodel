//! Full-mode, full-policy executable-oracle conformance matrix generation.

use super::{GeneratedPair, GeneratorError, PinnedOracle, SeedSchema, candidates_for_schema};
use crate::models::{
    InvariantEvidenceConfig, OracleCompatibilityMode, OracleConformanceReport, OracleConformanceRun,
};
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
                base_schema: seed.schema.clone(),
                candidate_schema: candidate.candidate_schema,
            });
        }
    }
    let runs = run_jobs_in_parallel(oracle, workspace, policies, modes, jobs)?;
    Ok(OracleConformanceReport::new(
        oracle.jar_sha256().to_owned(),
        oracle.policy_packs_sha256().to_owned(),
        thresholds,
        runs,
    ))
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
                                    compatibility_mode: mode,
                                    policy_pack: policy_pack.clone(),
                                    exit_code: run.exit_code,
                                    outcome: run.outcome.as_str().to_owned(),
                                    stdout: run.stdout,
                                    stderr: run.stderr,
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
