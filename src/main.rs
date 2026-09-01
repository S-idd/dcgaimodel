use dcgaimodel::dataset::DatasetSplitConfig;
use dcgaimodel::evaluation::{ClassificationMetrics, evaluate_binary_classification};
use dcgaimodel::generation::{
    CoverageTarget, GeneratorConfig, JsonSchemaBenchSource, MutationKind,
    OptionalProfileCoverageTarget, OracleConfig, PinnedOracle, SourceSamplingProfile,
    TrainingDataGenerator, build_oracle_conformance_matrix, has_fatal_oracle_runtime_failure,
    refeature_v5_as_v6,
};
use dcgaimodel::models::{
    ChallengeProtocol, DatasetRole, DcgPipelineConfig, DcgPipelineResult, ExternalEvaluationConfig,
    ExternalFeatureDiagnosticConfig, ExternalManifestPreflightConfig, InvariantEvidenceConfig,
    ModelArtifact, ModelConfig, OracleCompatibilityMode, OracleConformanceReport,
    OracleInvariantPromotionManifest, PreparedDcgDataset, TargetMode, ThreeWayExperimentConfig,
    TrainingInputProvenance, TrainingMetadata, audit_challenge_near_duplicates,
    diagnose_external_feature_space, evaluate_external_transitions, evaluate_three_way,
    preflight_external_manifest, run_three_way_compatibility_pipeline, run_three_way_experiment,
    structural_variant_key,
};
use dcgaimodel::nn::Sgd;
use dcgaimodel::prediction::PredictionKind;
use dcgaimodel::training::TrainingConfig;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let mut arguments = env::args().skip(1);
    let result = match arguments.next().as_deref() {
        Some("generate") => generate(arguments.collect()),
        Some("refeature-v6") => refeature_v6(arguments.collect()),
        Some("audit") => audit(arguments.collect()),
        Some("oracle-conformance") => oracle_conformance(arguments.collect()),
        Some("reclassify-conformance") => reclassify_conformance(arguments.collect()),
        Some("merge-conformance") => merge_conformance(arguments.collect()),
        Some("promote-invariants") => promote_invariants(arguments.collect()),
        Some("train") => train(arguments.collect()),
        Some("train-three-way") => train_three_way(arguments.collect()),
        Some("evaluate-three-way-mutation") => evaluate_three_way_mutation(arguments.collect()),
        Some("evaluate-protocols") => evaluate_protocols(arguments.collect()),
        Some("run-three-way-experiments") => run_three_way_experiments(arguments.collect()),
        Some("audit-challenge-near-duplicates") => {
            audit_challenge_near_duplicates_command(arguments.collect())
        }
        Some("evaluate-external") => evaluate_external(arguments.collect()),
        Some("preflight-external") => preflight_external(arguments.collect()),
        Some("diagnose-external-features") => diagnose_external_features(arguments.collect()),
        _ => Err("Usage: dcgaimodel <generate|refeature-v6|audit|oracle-conformance|reclassify-conformance|merge-conformance|promote-invariants|train|train-three-way|evaluate-three-way-mutation|evaluate-protocols|run-three-way-experiments|audit-challenge-near-duplicates|evaluate-external|preflight-external|diagnose-external-features> [options]".to_owned()),
    };
    if let Err(error) = result {
        eprintln!("dcgaimodel failed: {error}");
        std::process::exit(2);
    }
}

/// Writes a bounded, reproducible overlap audit for selected held-out
/// challenge protocols. This audit reports family/pair identity leakage and
/// model-visible feature overlap; raw schema text is not persisted in a
/// prepared corpus and therefore cannot be compared here.
fn audit_challenge_near_duplicates_command(arguments: Vec<String>) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut protocols = Vec::new();
    let mut max_samples = 5usize;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--protocol" => protocols.push(value.clone()),
            "--max-samples" => {
                max_samples = value.parse().map_err(|_| {
                    format!("--max-samples must be a positive whole number, got {value}")
                })?
            }
            _ => {
                return Err(format!(
                    "Unknown audit-challenge-near-duplicates option: {flag}"
                ));
            }
        }
        index += 2;
    }
    let input = input.ok_or("Missing --input")?;
    let output = output.ok_or("Missing --output")?;
    if output.exists() {
        return Err(format!(
            "--output already exists; refusing to overwrite {}",
            output.display()
        ));
    }
    let dataset = PreparedDcgDataset::load(&input).map_err(|error| error.to_string())?;
    let report = audit_challenge_near_duplicates(&dataset, &protocols, max_samples)?;
    let serialized = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(&output, serialized).map_err(|error| error.to_string())?;
    println!(
        "Challenge near-duplicate audit complete: protocols={}, output={}",
        report.protocols.len(),
        output.display()
    );
    for protocol in &report.protocols {
        println!(
            "{}: families_leak={}, pairs_leak={}, full_feature_overlap={}/{}, structural_overlap={}/{}, within_one={}/{}",
            protocol.protocol,
            protocol.family_leakage,
            protocol.pair_fingerprint_leakage,
            protocol
                .exact_model_feature_overlap
                .challenge_records_with_match,
            protocol.challenge_records,
            protocol
                .exact_policy_free_structural_overlap
                .challenge_records_with_match,
            protocol.challenge_records,
            protocol.challenge_records_within_one_structural_coordinate,
            protocol.challenge_records,
        );
    }
    Ok(())
}

/// Writes a no-oracle feature-space explanation for one external transition.
fn diagnose_external_features(arguments: Vec<String>) -> Result<(), String> {
    let mut manifest = None;
    let mut v9_dataset = None;
    let mut policy_packs = None;
    let mut record_id = None;
    let mut output = None;
    let mut nearest_record_limit = 5usize;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value)),
            "--v9-dataset" => v9_dataset = Some(PathBuf::from(value)),
            "--policy-packs" => policy_packs = Some(PathBuf::from(value)),
            "--record-id" => record_id = Some(value.clone()),
            "--output" => output = Some(PathBuf::from(value)),
            "--nearest-record-limit" => {
                nearest_record_limit = value.parse().map_err(|_| {
                    format!("--nearest-record-limit must be a positive whole number, got {value}")
                })?
            }
            _ => return Err(format!("Unknown diagnose-external-features option: {flag}")),
        }
        index += 2;
    }
    let output = output.ok_or("Missing --output")?;
    if output.exists() {
        return Err(format!(
            "--output already exists; refusing to overwrite {}",
            output.display()
        ));
    }
    let report = diagnose_external_feature_space(&ExternalFeatureDiagnosticConfig {
        manifest_path: manifest.ok_or("Missing --manifest")?,
        v9_dataset_path: v9_dataset.ok_or("Missing --v9-dataset")?,
        policy_packs_path: policy_packs.ok_or("Missing --policy-packs")?,
        record_id: record_id.ok_or("Missing --record-id")?,
        nearest_record_limit,
    })?;
    let serialized = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(&output, serialized).map_err(|error| error.to_string())?;
    println!(
        "External feature diagnostic complete: record={}, nearest_distance={}, output={}",
        report.record_id,
        report.nearest_v9_structural_distance,
        output.display()
    );
    Ok(())
}

/// Validates external input and applies V9 overlap gates without launching
/// Java, applying a scaler, or running a model.
fn preflight_external(arguments: Vec<String>) -> Result<(), String> {
    let mut manifest = None;
    let mut v9_dataset = None;
    let mut jar = None;
    let mut policy_packs = None;
    let mut output = None;
    let mut max_structural_distance = 1usize;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value)),
            "--v9-dataset" => v9_dataset = Some(PathBuf::from(value)),
            "--jar" => jar = Some(PathBuf::from(value)),
            "--policy-packs" => policy_packs = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--max-structural-distance" => {
                max_structural_distance = value.parse().map_err(|_| {
                    format!(
                        "--max-structural-distance must be a non-negative whole number, got {value}"
                    )
                })?
            }
            _ => return Err(format!("Unknown preflight-external option: {flag}")),
        }
        index += 2;
    }
    let output = output.ok_or("Missing --output")?;
    if output.exists() {
        return Err(format!(
            "--output already exists; refusing to overwrite {}",
            output.display()
        ));
    }
    let report = preflight_external_manifest(&ExternalManifestPreflightConfig {
        manifest_path: manifest.ok_or("Missing --manifest")?,
        v9_dataset_path: v9_dataset.ok_or("Missing --v9-dataset")?,
        jar_path: jar.ok_or("Missing --jar")?,
        policy_packs_path: policy_packs.ok_or("Missing --policy-packs")?,
        max_structural_coordinate_distance: max_structural_distance,
    })?;
    let serialized = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(&output, serialized).map_err(|error| error.to_string())?;
    println!(
        "External manifest preflight complete: passed={}, total={}, accepted={}, rejected={}, output={}",
        report.passed,
        report.total_records,
        report.accepted_records,
        report.rejected_records,
        output.display(),
    );
    Ok(())
}

/// Labels externally sourced transitions with the pinned oracle, excludes all
/// V9 identity/near-feature overlap, and runs frozen three-way inference only.
/// It cannot train a network, refit a scaler, or modify either V9 artifact.
fn evaluate_external(arguments: Vec<String>) -> Result<(), String> {
    let mut manifest = None;
    let mut v9_dataset = None;
    let mut model = None;
    let mut jar = None;
    let mut policy_packs = None;
    let mut workspace = None;
    let mut output = None;
    let mut max_structural_distance = 1usize;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value)),
            "--v9-dataset" => v9_dataset = Some(PathBuf::from(value)),
            "--model" => model = Some(PathBuf::from(value)),
            "--jar" => jar = Some(PathBuf::from(value)),
            "--policy-packs" => policy_packs = Some(PathBuf::from(value)),
            "--workspace" => workspace = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--max-structural-distance" => {
                max_structural_distance = value.parse().map_err(|_| {
                    format!(
                        "--max-structural-distance must be a non-negative whole number, got {value}"
                    )
                })?
            }
            _ => return Err(format!("Unknown evaluate-external option: {flag}")),
        }
        index += 2;
    }
    let output = output.ok_or("Missing --output")?;
    if output.exists() {
        return Err(format!(
            "--output already exists; refusing to overwrite {}",
            output.display()
        ));
    }
    let config = ExternalEvaluationConfig {
        manifest_path: manifest.ok_or("Missing --manifest")?,
        v9_dataset_path: v9_dataset.ok_or("Missing --v9-dataset")?,
        model_path: model.ok_or("Missing --model")?,
        oracle_config: OracleConfig {
            java_program: PathBuf::from("java"),
            jar_path: jar.ok_or("Missing --jar")?,
            policy_packs_path: policy_packs.ok_or("Missing --policy-packs")?,
        },
        oracle_workspace: workspace.ok_or("Missing --workspace")?,
        max_structural_coordinate_distance: max_structural_distance,
    };
    let report = evaluate_external_transitions(&config)?;
    let serialized = serde_json::to_string_pretty(&report).map_err(|error| error.to_string())?;
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(&output, serialized).map_err(|error| error.to_string())?;
    println!(
        "Frozen external evaluation complete: total={}, scored={}, oracle_rejected={}, overlap_rejected={}, output={}",
        report.total_manifest_records,
        report.scored_records,
        report.oracle_rejected_records,
        report.overlap_rejected_records,
        output.display(),
    );
    println!(
        "overall: records={}, accuracy={:?}, confusion={:?}",
        report.overall.records, report.overall.accuracy, report.overall.confusion_matrix
    );
    for (mutation, metrics) in &report.by_mutation {
        println!(
            "mutation {mutation}: records={}, accuracy={:?}, confusion={:?}",
            metrics.records, metrics.accuracy, metrics.confusion_matrix
        );
    }
    for (policy, metrics) in &report.by_policy_pack {
        println!(
            "policy {policy}: records={}, accuracy={:?}, confusion={:?}",
            metrics.records, metrics.accuracy, metrics.confusion_matrix
        );
    }
    Ok(())
}

/// Rebuilds label-free V6 features for an existing V5 artifact after proving
/// every persisted pair still matches its source/schema/policy/JAR identity.
/// This command intentionally never invokes the oracle executable.
fn refeature_v6(arguments: Vec<String>) -> Result<(), String> {
    let mut input = None;
    let mut source_path = None;
    let mut source_name = "jsonschemabench".to_owned();
    let mut max_source_schemas = None;
    let mut policy_packs = None;
    let mut oracle_jar = None;
    let mut output = None;
    let mut dataset_version = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--source" => source_path = Some(PathBuf::from(value)),
            "--source-name" => source_name = value.to_owned(),
            "--max-source-schemas" => {
                max_source_schemas = Some(value.parse::<usize>().map_err(|_| {
                    format!("--max-source-schemas must be a positive whole number, got {value}")
                })?)
            }
            "--policy-packs" => policy_packs = Some(PathBuf::from(value)),
            "--oracle-jar" => oracle_jar = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--dataset-version" => dataset_version = Some(value.to_owned()),
            _ => return Err(format!("Unknown refeature-v6 option: {flag}")),
        }
        index += 2;
    }
    let input = input.ok_or("Missing --input")?;
    let source_path = source_path.ok_or("Missing --source")?;
    let max_source_schemas = max_source_schemas.ok_or("Missing --max-source-schemas")?;
    if max_source_schemas == 0 {
        return Err("--max-source-schemas must be positive".to_owned());
    }
    let policy_packs = policy_packs.ok_or("Missing --policy-packs")?;
    let oracle_jar = oracle_jar.ok_or("Missing --oracle-jar")?;
    let output = output.ok_or("Missing --output")?;
    let dataset_version = dataset_version.ok_or("Missing --dataset-version")?;
    if output.exists() {
        return Err(format!(
            "--output already exists; choose a new artifact path: {}",
            output.display()
        ));
    }
    let dataset = PreparedDcgDataset::load(&input).map_err(|error| error.to_string())?;
    let source = JsonSchemaBenchSource {
        path: source_path,
        source: source_name,
    };
    let (migrated, report) = refeature_v5_as_v6(
        &dataset,
        &source,
        max_source_schemas,
        &policy_packs,
        &oracle_jar,
        dataset_version,
    )
    .map_err(|error| error.to_string())?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    migrated.save(&output).map_err(|error| error.to_string())?;
    let persisted = PreparedDcgDataset::load(&output).map_err(|error| error.to_string())?;
    println!(
        "V6 feature migration complete: records={}, families={}, feature_version={}, generation_seed={}, sampled_source_families={}, verified_pair_fingerprints={}, output={}",
        persisted.len(),
        persisted.statistics().independent_families,
        persisted.feature_version(),
        report.generation_seed,
        report.sampled_source_families,
        report.verified_records,
        output.display(),
    );
    Ok(())
}

fn run_three_way_experiments(arguments: Vec<String>) -> Result<(), String> {
    let mut input = None;
    let mut output_dir = None;
    let mut oracle_jar = None;
    let mut policy_packs = None;
    let mut epochs = 100usize;
    let mut batch_size = 16usize;
    let mut learning_rate = 0.05f64;
    let mut seeds = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--output-dir" => output_dir = Some(PathBuf::from(value)),
            "--oracle-jar" => oracle_jar = Some(PathBuf::from(value)),
            "--policy-packs" => policy_packs = Some(PathBuf::from(value)),
            "--epochs" => {
                epochs = value
                    .parse()
                    .map_err(|_| format!("--epochs must be a whole number, got {value}"))?
            }
            "--batch-size" => {
                batch_size = value
                    .parse()
                    .map_err(|_| format!("--batch-size must be a whole number, got {value}"))?
            }
            "--learning-rate" => {
                learning_rate = value
                    .parse()
                    .map_err(|_| format!("--learning-rate must be a finite number, got {value}"))?
            }
            "--seed" => seeds.push(
                value
                    .parse()
                    .map_err(|_| format!("--seed must be a whole number, got {value}"))?,
            ),
            "--seeds" => {
                let parsed = value
                    .split(',')
                    .map(str::trim)
                    .filter(|seed| !seed.is_empty())
                    .map(|seed| {
                        seed.parse::<u64>().map_err(|_| {
                            format!("--seeds contains an invalid whole number: {seed}")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if parsed.is_empty() {
                    return Err("--seeds must contain at least one whole number".to_owned());
                }
                seeds.extend(parsed);
            }
            _ => return Err(format!("Unknown run-three-way-experiments option: {flag}")),
        }
        index += 2;
    }
    let input = input.ok_or("Missing --input")?;
    let output_dir = output_dir.ok_or("Missing --output-dir")?;
    let oracle_jar = oracle_jar.ok_or("Missing --oracle-jar")?;
    let policy_packs = policy_packs.ok_or("Missing --policy-packs")?;
    if seeds.is_empty() {
        // Stable default seeds make the ordinary command a multi-seed run.
        seeds = vec![20_260_826, 20_260_827, 20_260_828];
    }
    let configuration = ThreeWayExperimentConfig::new(seeds, epochs, batch_size, learning_rate)?;
    let dataset = PreparedDcgDataset::load(&input).map_err(|error| error.to_string())?;
    let input_provenance =
        verify_experiment_input_provenance(&dataset, &input, &oracle_jar, &policy_packs)?;
    ensure_fresh_output_directory(&output_dir)?;
    fs::create_dir_all(&output_dir).map_err(|error| error.to_string())?;
    let report = run_three_way_experiment(&dataset, input_provenance, configuration, &output_dir)?;
    let report_path = output_dir.join("three-way-experiment-report.json");
    report.save(&report_path)?;
    println!(
        "Three-way CPU experiment complete: seeds={:?}, models={}, report={}",
        report.configuration.seeds,
        report
            .seed_runs
            .iter()
            .flat_map(|seed| seed.protocols.iter())
            .filter(|protocol| protocol.model_artifact.is_some())
            .count(),
        report_path.display(),
    );
    for aggregate in &report.aggregate_protocols {
        println!(
            "aggregate {}: runs={}, mean_accuracy={:.6}, range=[{:.6}, {:.6}], confusion={:?}",
            aggregate.protocol,
            aggregate.scored_runs,
            aggregate.mean_accuracy,
            aggregate.minimum_accuracy,
            aggregate.maximum_accuracy,
            aggregate.summed_confusion_matrix,
        );
    }
    Ok(())
}

fn ensure_fresh_output_directory(path: &Path) -> Result<(), String> {
    if path.exists() {
        if !path.is_dir() {
            return Err(format!(
                "--output-dir is not a directory: {}",
                path.display()
            ));
        }
        if fs::read_dir(path)
            .map_err(|error| error.to_string())?
            .next()
            .is_some()
        {
            return Err(format!(
                "--output-dir must be empty to prevent overwriting an experiment: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn verify_experiment_input_provenance(
    dataset: &PreparedDcgDataset,
    dataset_path: &Path,
    oracle_jar: &Path,
    policy_packs: &Path,
) -> Result<TrainingInputProvenance, String> {
    let dataset_sha256 = sha256_file(dataset_path)?;
    let oracle_jar_sha256 = sha256_file(oracle_jar)?;
    let policy_packs_sha256 = sha256_file(policy_packs)?;
    let dataset_jar_hashes = dataset
        .records()
        .iter()
        .filter_map(|record| {
            record
                .generation
                .as_ref()
                .map(|generation| generation.oracle_jar_sha256.clone())
        })
        .collect::<BTreeSet<_>>();
    let dataset_policy_hashes = dataset
        .records()
        .iter()
        .filter_map(|record| {
            record
                .generation
                .as_ref()
                .map(|generation| generation.policy_packs_sha256.clone())
        })
        .collect::<BTreeSet<_>>();
    if dataset_jar_hashes.len() != 1 || dataset_policy_hashes.len() != 1 {
        return Err(format!(
            "prepared dataset must have exactly one generated oracle and policy-pack identity; found jar={dataset_jar_hashes:?}, policies={dataset_policy_hashes:?}"
        ));
    }
    if dataset_jar_hashes.first() != Some(&oracle_jar_sha256) {
        return Err(format!(
            "--oracle-jar SHA-256 does not match the dataset's oracle evidence: expected {:?}, got {oracle_jar_sha256}",
            dataset_jar_hashes.first()
        ));
    }
    if dataset_policy_hashes.first() != Some(&policy_packs_sha256) {
        return Err(format!(
            "--policy-packs SHA-256 does not match the dataset's oracle evidence: expected {:?}, got {policy_packs_sha256}",
            dataset_policy_hashes.first()
        ));
    }
    TrainingInputProvenance {
        dataset_sha256,
        oracle_jar_sha256,
        policy_packs_sha256,
    }
    .validate()
    .map_err(|error| error.to_string())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let contents = fs::read(path).map_err(|error| {
        format!(
            "could not read pinned input {}: {error}",
            path.to_string_lossy()
        )
    })?;
    Ok(format!("{:x}", Sha256::digest(contents)))
}

fn audit(arguments: Vec<String>) -> Result<(), String> {
    let mut input = None;
    let mut promotions = None;
    let mut seed = 42u64;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--promotions" => promotions = Some(PathBuf::from(value)),
            "--seed" => {
                seed = value
                    .parse()
                    .map_err(|_| format!("--seed must be a whole number, got {value}"))?
            }
            _ => return Err(format!("Unknown audit option: {flag}")),
        }
        index += 2;
    }
    let input = input.ok_or("Missing --input")?;
    let dataset = PreparedDcgDataset::load(&input).map_err(|error| error.to_string())?;
    let split_config =
        DatasetSplitConfig::new(0.70, 0.15, 0.15, seed).map_err(|error| error.to_string())?;
    let binary = dataset.training_readiness(TargetMode::BinaryBreaking, split_config);
    let three_way = dataset.training_readiness(TargetMode::ThreeWayCompatibility, split_config);
    let promotion_manifest = promotions
        .map(|path| OracleInvariantPromotionManifest::load(&path))
        .transpose()?;
    let generalization = promotion_manifest.as_ref().map_or_else(
        || dataset.generalization_readiness(),
        |manifest| dataset.generalization_readiness_with_promotions(manifest),
    );
    let challenge_families = dataset
        .records()
        .iter()
        .filter(|record| record.dataset_role == DatasetRole::Challenge)
        .map(|record| record.family_id.as_str())
        .collect::<BTreeSet<_>>();
    let challenge_records = dataset
        .records()
        .iter()
        .filter(|record| record.dataset_role == DatasetRole::Challenge)
        .count();
    println!(
        "Dataset audit: version={}, feature_version={}, records={}, families={}, challenge_records={}, challenge_families={}, binary_ready={}, three_way_ready={}",
        dataset.dataset_version(),
        dataset.feature_version(),
        dataset.len(),
        binary.statistics.independent_families,
        challenge_records,
        challenge_families.len(),
        binary.ready_for_training,
        three_way.ready_for_training,
    );
    match dataset.split_by_group(split_config) {
        Ok(split) => {
            for (name, partition) in [
                ("train", split.train()),
                ("validation", split.validation()),
                ("test", split.test()),
            ] {
                let mut labels = BTreeMap::<&str, usize>::new();
                let mut policies = BTreeMap::<&str, usize>::new();
                let mut mutations = BTreeMap::<&str, usize>::new();
                let mut families = BTreeSet::new();
                for record in partition.records() {
                    *labels
                        .entry(
                            record
                                .compatibility_label
                                .map_or("missing", |label| label.as_str()),
                        )
                        .or_default() += 1;
                    *policies.entry(&record.policy_pack).or_default() += 1;
                    if let Some(generation) = &record.generation {
                        *mutations.entry(&generation.declared_mutation).or_default() += 1;
                    }
                    families.insert(&record.split_group_id);
                }
                println!(
                    "{name}: records={}, families={}, labels={labels:?}, policies={policies:?}, mutations={mutations:?}",
                    partition.len(),
                    families.len(),
                );
            }
        }
        Err(error) => println!("Normal split unavailable: {error}"),
    }
    println!(
        "Shortcut audit: policy×label={:?}",
        generalization.policy_by_label
    );
    println!(
        "Shortcut audit: mutation×label={:?}",
        generalization.mutation_by_label
    );
    println!(
        "Shortcut audit: policy×mutation×label={:?}",
        generalization.policy_mutation_by_label
    );
    println!(
        "Counterfactual audit: family_mutation_pairs={}, complete_three_policy_pairs={}, observed_oracle_invariant_mutations={:?}, shortcut_risks={:?}",
        generalization.family_mutation_pairs,
        generalization.complete_policy_counterfactual_pairs,
        generalization.observed_oracle_invariant_mutations,
        generalization.shortcut_risks,
    );
    println!(
        "Optional-field structural coverage: {:?}",
        generalization.optional_field_profile_coverage
    );
    for exception in &generalization.oracle_invariant_exceptions {
        println!(
            "oracle-invariant mutation {}: policies={:?}, labels={:?}, rationale={}",
            exception.mutation,
            exception.observed_policies,
            exception.observed_labels,
            exception.rationale,
        );
    }
    println!(
        "Challenge audit: standard_families={}, family_leakage={}, pair_leakage={}, benchmark_ready={}, reasons={:?}",
        generalization.standard_families,
        generalization.challenge_family_leakage,
        generalization.challenge_pair_leakage,
        generalization.ready_for_generalization_benchmark,
        generalization.reasons,
    );
    for protocol in generalization.challenge_protocols {
        println!(
            "challenge {}: records={}, families={}, policies={}, mutations={}, labels={:?}, meaningful={}, reason={:?}",
            protocol.protocol,
            protocol.records,
            protocol.families,
            protocol.policies,
            protocol.mutations,
            protocol.labels,
            protocol.meaningful,
            protocol.reason,
        );
    }
    Ok(())
}

fn oracle_conformance(arguments: Vec<String>) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut jar = None;
    let mut policy_packs = None;
    let mut workspace = None;
    let mut policies = Vec::new();
    let mut source_profile = SourceSamplingProfile::Broad;
    let mut compatibility_modes = Vec::new();
    let mut mutation_families = Vec::new();
    let mut max = 50usize;
    let mut seed = 42u64;
    let mut min_families = 25usize;
    let mut min_policies = 3usize;
    let mut min_checks = 75usize;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--jar" => jar = Some(PathBuf::from(value)),
            "--policy-packs" => policy_packs = Some(PathBuf::from(value)),
            "--workspace" => workspace = Some(PathBuf::from(value)),
            "--policy" => policies.push(value.clone()),
            "--source-profile" => {
                source_profile = SourceSamplingProfile::parse(value).ok_or_else(|| {
                    format!(
                        "--source-profile must be broad, optional-open, or optional-closed, got {value}"
                    )
                })?
            }
            "--mode" => compatibility_modes.push(
                OracleCompatibilityMode::parse(value).ok_or_else(|| {
                    format!("--mode must be BACKWARD, FORWARD, or FULL, got {value}")
                })?,
            ),
            "--mutation" => {
                let mutation = MutationKind::parse(value)
                    .ok_or_else(|| format!("Unknown mutation family for --mutation: {value}"))?;
                mutation_families.push(mutation.as_str().to_owned());
            }
            "--max" => {
                max = value
                    .parse()
                    .map_err(|_| format!("--max must be a whole number, got {value}"))?
            }
            "--seed" => {
                seed = value
                    .parse()
                    .map_err(|_| format!("--seed must be a whole number, got {value}"))?
            }
            "--min-families" => {
                min_families = value
                    .parse()
                    .map_err(|_| format!("--min-families must be a whole number, got {value}"))?
            }
            "--min-policies" => {
                min_policies = value
                    .parse()
                    .map_err(|_| format!("--min-policies must be a whole number, got {value}"))?
            }
            "--min-checks" => {
                min_checks = value
                    .parse()
                    .map_err(|_| format!("--min-checks must be a whole number, got {value}"))?
            }
            _ => return Err(format!("Unknown oracle-conformance option: {flag}")),
        }
        index += 2;
    }
    let (input, output, jar, policy_packs, workspace) = (
        input.ok_or("Missing --input")?,
        output.ok_or("Missing --output")?,
        jar.ok_or("Missing --jar")?,
        policy_packs.ok_or("Missing --policy-packs")?,
        workspace.ok_or("Missing --workspace")?,
    );
    if max == 0 || min_families == 0 || min_policies == 0 || min_checks == 0 {
        return Err("conformance limits must be positive".to_owned());
    }
    if policies.is_empty() {
        policies = approved_policy_names(&policy_packs)?;
    }
    if compatibility_modes.is_empty() {
        compatibility_modes = OracleCompatibilityMode::all().to_vec();
    }
    compatibility_modes.sort_unstable();
    compatibility_modes.dedup();
    let source = JsonSchemaBenchSource {
        path: input,
        source: "jsonschemabench-conformance".to_owned(),
    };
    let seeds = source
        .load_sampled_with_profile(max, seed, source_profile)
        .map_err(|error| error.to_string())?;
    let oracle = PinnedOracle::new(OracleConfig {
        java_program: PathBuf::from("java"),
        jar_path: jar,
        policy_packs_path: policy_packs,
    })
    .map_err(|error| error.to_string())?;
    let report = build_oracle_conformance_matrix(
        &oracle,
        &workspace,
        &policies,
        &compatibility_modes,
        &mutation_families,
        InvariantEvidenceConfig {
            min_independent_families: min_families,
            min_policies,
            min_oracle_checks: min_checks,
        },
        seeds,
    )
    .map_err(|error| error.to_string())?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    report.save(&output)?;
    println!(
        "Oracle conformance: runs={}, cells={}, cross_policy={}, jar_sha256={}, policy_packs_sha256={}, output={}",
        report.runs.len(),
        report.cells.len(),
        report.cross_policy.len(),
        report.oracle_jar_sha256,
        report.policy_packs_sha256,
        output.display()
    );
    for evidence in &report.cross_policy {
        println!(
            "{} {}: families={}, policies={:?}, accepted_policies={:?}, checks={}, accepted_checks={}, accepted_labels={:?}, all_outcomes={:?}, candidate={}",
            evidence.mutation_family,
            evidence.compatibility_mode.as_str(),
            evidence.independent_families,
            evidence.policies_checked,
            evidence.accepted_policies_checked,
            evidence.oracle_checks,
            evidence.accepted_oracle_checks,
            evidence.accepted_labels_observed,
            evidence.labels_observed,
            evidence.candidate_invariant
        );
    }
    Ok(())
}

/// Returns every declared approved policy name from the immutable pack file.
/// Conformance defaults to this complete set: silently auditing only the
/// historical baseline/strict/relaxed trio would make an expanded approved
/// V5/V6 pack look more complete than it is.
fn approved_policy_names(policy_packs: &Path) -> Result<Vec<String>, String> {
    let document = fs::read_to_string(policy_packs).map_err(|error| error.to_string())?;
    let parsed =
        serde_json::from_str::<serde_json::Value>(&document).map_err(|error| error.to_string())?;
    let packs = parsed
        .get("packs")
        .and_then(serde_json::Value::as_object)
        .ok_or("policy-pack file must contain an object `packs`")?;
    let names = packs.keys().cloned().collect::<Vec<_>>();
    if names.is_empty() {
        return Err("policy-pack file contains no approved packs".to_owned());
    }
    Ok(names)
}

/// Rebuilds derived conformance evidence after rejecting raw exit-1 runs that
/// contain an uncaught JVM exception rather than a compatibility decision.
/// Raw JAR output, hashes, modes, policies, and non-fatal outcomes are kept
/// exactly as captured; this command never invokes the oracle or invents a
/// new label.
fn reclassify_conformance(arguments: Vec<String>) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("Unknown reclassify-conformance option: {flag}")),
        }
        index += 2;
    }
    let input = input.ok_or("Missing --input")?;
    let output = output.ok_or("Missing --output")?;
    if output.exists() {
        return Err(format!(
            "--output already exists; choose a new evidence path: {}",
            output.display()
        ));
    }
    let mut report = OracleConformanceReport::load(&input)?;
    let mut corrected = 0usize;
    for run in &mut report.runs {
        if run.exit_code == 1 && has_fatal_oracle_runtime_failure(&run.stdout, &run.stderr) {
            run.outcome = "rejected".to_owned();
            corrected += 1;
        }
    }
    report.rebuild_aggregates();
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    report.save(&output)?;
    println!(
        "Reclassified conformance: fatal_exit_one_rejected={}, runs={}, jar_sha256={}, policy_packs_sha256={}, output={}",
        corrected,
        report.runs.len(),
        report.oracle_jar_sha256,
        report.policy_packs_sha256,
        output.display()
    );
    Ok(())
}

/// Combines independent, hash-identical conformance samples into one reviewable
/// evidence artifact. It preserves every raw invocation and refuses duplicate
/// family/mutation/mode/policy cells rather than silently hiding evidence.
fn merge_conformance(arguments: Vec<String>) -> Result<(), String> {
    let mut inputs = Vec::new();
    let mut output = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => inputs.push(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("Unknown merge-conformance option: {flag}")),
        }
        index += 2;
    }
    if inputs.len() < 2 {
        return Err("merge-conformance requires at least two --input reports".to_owned());
    }
    let output = output.ok_or("Missing --output")?;
    if output.exists() {
        return Err(format!(
            "--output already exists; choose a new evidence path: {}",
            output.display()
        ));
    }

    let first_input = &inputs[0];
    let first = OracleConformanceReport::load(first_input)?;
    let jar_hash = first.oracle_jar_sha256.clone();
    let policy_hash = first.policy_packs_sha256.clone();
    let thresholds = first.thresholds.clone();
    let mut runs = first.runs;
    let mut seen = runs
        .iter()
        .map(|run| {
            (
                run.family_id.clone(),
                run.mutation_family.clone(),
                run.compatibility_mode,
                run.policy_pack.clone(),
            )
        })
        .collect::<BTreeSet<_>>();

    for input in inputs.iter().skip(1) {
        let report = OracleConformanceReport::load(input)?;
        if report.oracle_jar_sha256 != jar_hash
            || report.policy_packs_sha256 != policy_hash
            || report.thresholds != thresholds
        {
            return Err(format!(
                "cannot merge {}: JAR hash, policy-pack hash, or evidence thresholds differ from {}",
                input.display(),
                first_input.display()
            ));
        }
        for run in report.runs {
            let identity = (
                run.family_id.clone(),
                run.mutation_family.clone(),
                run.compatibility_mode,
                run.policy_pack.clone(),
            );
            if !seen.insert(identity) {
                return Err(format!(
                    "cannot merge {}: duplicate family/mutation/mode/policy evidence cell",
                    input.display()
                ));
            }
            runs.push(run);
        }
    }

    let merged = OracleConformanceReport::new(jar_hash, policy_hash, thresholds, runs);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    merged.save(&output)?;
    println!(
        "Merged conformance: reports={}, runs={}, cells={}, cross_policy={}, jar_sha256={}, policy_packs_sha256={}, output={}",
        inputs.len(),
        merged.runs.len(),
        merged.cells.len(),
        merged.cross_policy.len(),
        merged.oracle_jar_sha256,
        merged.policy_packs_sha256,
        output.display()
    );
    Ok(())
}

fn promote_invariants(arguments: Vec<String>) -> Result<(), String> {
    let mut evidence = None;
    let mut output = None;
    let mut existing = None;
    let mut selections = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--evidence" => evidence = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--existing" => existing = Some(PathBuf::from(value)),
            "--select" => {
                let (key, rationale) = value
                    .split_once('=')
                    .ok_or("--select must use mutation:MODE:policy=rationale")?;
                let mut parts = key.split(':');
                let (Some(mutation), Some(mode), Some(policy), None) =
                    (parts.next(), parts.next(), parts.next(), parts.next())
                else {
                    return Err("--select must use mutation:MODE:policy=rationale".to_owned());
                };
                let mode = OracleCompatibilityMode::parse(mode)
                    .ok_or("--select mode must be BACKWARD, FORWARD, or FULL")?;
                selections.push((
                    mutation.to_owned(),
                    mode,
                    policy.to_owned(),
                    rationale.to_owned(),
                ));
            }
            _ => return Err(format!("Unknown promote-invariants option: {flag}")),
        }
        index += 2;
    }
    let report = OracleConformanceReport::load(&evidence.ok_or("Missing --evidence")?)?;
    let manifest = report.promote(&selections)?;
    let manifest = if let Some(existing) = existing {
        OracleInvariantPromotionManifest::load(&existing)?.merge(manifest)?
    } else {
        manifest
    };
    let output = output.ok_or("Missing --output")?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    manifest.save(&output)?;
    println!(
        "Promoted {} qualified oracle invariants: {}",
        manifest.promotions.len(),
        output.display()
    );
    Ok(())
}

fn generate(arguments: Vec<String>) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut jar = None;
    let mut policy_packs = None;
    let mut workspace = None;
    let mut source = "jsonschemabench".to_owned();
    let mut policies = Vec::new();
    let mut source_profile = SourceSamplingProfile::Broad;
    let mut compatibility_mode = OracleCompatibilityMode::Backward;
    let mut mutation_families = Vec::new();
    let mut retain_variant_families = Vec::new();
    let mut max = 100usize;
    let mut seed = 42u64;
    let mut oracle_workers = 1usize;
    let mut per_outcome = None;
    let mut per_policy_mutation_outcome = None;
    let mut challenge_family_ratio = 0.15f64;
    let mut target_records = None;
    let mut target_standard_families = None;
    let mut target_challenge_families = None;
    let mut target_counterfactual_pairs = None;
    let mut optional_profile_standard_families = None;
    let mut optional_profile_challenge_families = None;
    let mut promotions = None;
    let mut dataset_version = "dcg-oracle-generated-v1".to_owned();
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--jar" => jar = Some(PathBuf::from(value)),
            "--policy-packs" => policy_packs = Some(PathBuf::from(value)),
            "--workspace" => workspace = Some(PathBuf::from(value)),
            "--source" => source = value.clone(),
            "--policy" => policies.push(value.clone()),
            "--source-profile" => {
                source_profile = SourceSamplingProfile::parse(value).ok_or_else(|| {
                    format!(
                        "--source-profile must be broad, optional-open, optional-closed, or balanced-optional, got {value}"
                    )
                })?
            }
            "--mode" => {
                compatibility_mode = OracleCompatibilityMode::parse(value).ok_or_else(|| {
                    format!("--mode must be BACKWARD, FORWARD, or FULL, got {value}")
                })?
            }
            "--mutation" => {
                let mutation = MutationKind::parse(value)
                    .ok_or_else(|| format!("Unknown mutation family for --mutation: {value}"))?;
                mutation_families.push(mutation.as_str().to_owned());
            }
            "--retain-variants-for" => {
                let mutation = MutationKind::parse(value).ok_or_else(|| {
                    format!("Unknown mutation family for --retain-variants-for: {value}")
                })?;
                retain_variant_families.push(mutation.as_str().to_owned());
            }
            "--max" => {
                max = value
                    .parse::<usize>()
                    .map_err(|_| format!("--max must be a positive whole number, got {value}"))?
            }
            "--seed" => {
                seed = value
                    .parse()
                    .map_err(|_| format!("--seed must be a whole number, got {value}"))?
            }
            "--oracle-workers" => {
                oracle_workers = value.parse().map_err(|_| {
                    format!("--oracle-workers must be a positive whole number, got {value}")
                })?
            }
            "--per-outcome" => {
                per_outcome = Some(value.parse::<usize>().map_err(|_| {
                    format!("--per-outcome must be a positive whole number, got {value}")
                })?)
            }
            "--per-policy-mutation-outcome" => {
                per_policy_mutation_outcome = Some(value.parse::<usize>().map_err(|_| {
                    format!(
                        "--per-policy-mutation-outcome must be a positive whole number, got {value}"
                    )
                })?)
            }
            "--challenge-family-ratio" => {
                challenge_family_ratio = value.parse().map_err(|_| {
                    format!("--challenge-family-ratio must be in [0, 1), got {value}")
                })?
            }
            "--target-records" => {
                target_records = Some(value.parse::<usize>().map_err(|_| {
                    format!("--target-records must be a positive whole number, got {value}")
                })?)
            }
            "--target-standard-families" => {
                target_standard_families = Some(value.parse::<usize>().map_err(|_| {
                    format!("--target-standard-families must be a whole number, got {value}")
                })?)
            }
            "--target-challenge-families" => {
                target_challenge_families = Some(value.parse::<usize>().map_err(|_| {
                    format!(
                        "--target-challenge-families must be a positive whole number, got {value}"
                    )
                })?)
            }
            "--target-counterfactual-pairs" => {
                target_counterfactual_pairs = Some(value.parse::<usize>().map_err(|_| {
                    format!(
                        "--target-counterfactual-pairs must be a positive whole number, got {value}"
                    )
                })?)
            }
            "--min-optional-profile-standard-families" => {
                optional_profile_standard_families = Some(value.parse::<usize>().map_err(|_| {
                    format!(
                        "--min-optional-profile-standard-families must be a positive whole number, got {value}"
                    )
                })?)
            }
            "--min-optional-profile-challenge-families" => {
                optional_profile_challenge_families = Some(value.parse::<usize>().map_err(|_| {
                    format!(
                        "--min-optional-profile-challenge-families must be a positive whole number, got {value}"
                    )
                })?)
            }
            "--promotions" => promotions = Some(PathBuf::from(value)),
            "--dataset-version" => dataset_version = value.clone(),
            _ => return Err(format!("Unknown generate option: {flag}")),
        }
        index += 2;
    }
    let input = input.ok_or("Missing --input")?;
    let output = output.ok_or("Missing --output")?;
    let jar = jar.ok_or("Missing --jar")?;
    let policy_packs = policy_packs.ok_or("Missing --policy-packs")?;
    let workspace = workspace.ok_or("Missing --workspace")?;
    if policies.is_empty() {
        policies = vec![
            "baseline".to_owned(),
            "strict".to_owned(),
            "relaxed".to_owned(),
        ];
    }
    let source = JsonSchemaBenchSource {
        path: input,
        source,
    };
    let mut config = GeneratorConfig::new(dataset_version, policies, max)
        .map_err(|error| error.to_string())?
        .with_seed(seed)
        .with_compatibility_mode(compatibility_mode);
    config = config
        .with_oracle_workers(oracle_workers)
        .map_err(|error| error.to_string())?;
    if !mutation_families.is_empty() {
        config = config
            .with_mutation_filter(mutation_families)
            .map_err(|error| error.to_string())?;
    }
    if !retain_variant_families.is_empty() {
        config = config
            .with_variant_retention(retain_variant_families)
            .map_err(|error| error.to_string())?;
    }
    if let Some(limit) = per_outcome {
        config = config
            .with_outcome_balance(limit)
            .map_err(|error| error.to_string())?;
    }
    if let Some(limit) = per_policy_mutation_outcome {
        config = config
            .with_policy_mutation_outcome_balance(limit)
            .map_err(|error| error.to_string())?;
    }
    config = config
        .with_challenge_family_ratio(challenge_family_ratio)
        .map_err(|error| error.to_string())?;
    if let Some(path) = promotions {
        config = config.with_promotion_manifest(OracleInvariantPromotionManifest::load(&path)?);
    }
    let target_values = [
        target_records,
        target_standard_families,
        target_challenge_families,
        target_counterfactual_pairs,
    ];
    if target_values.iter().any(Option::is_some) {
        let [
            Some(records),
            Some(standard_families),
            Some(challenge_families),
            Some(counterfactual_pairs),
        ] = target_values
        else {
            return Err("Coverage-driven generation requires --target-records, --target-standard-families, --target-challenge-families, and --target-counterfactual-pairs together".to_owned());
        };
        config = config.with_coverage_target(
            CoverageTarget::new(
                records,
                standard_families,
                challenge_families,
                counterfactual_pairs,
            )
            .map_err(|error| error.to_string())?,
        );
    }
    match (
        optional_profile_standard_families,
        optional_profile_challenge_families,
    ) {
        (None, None) => {}
        (Some(standard_families), Some(challenge_families)) => {
            if config.coverage_target.is_none() {
                return Err(
                    "optional root-profile coverage requires the four --target-* coverage options"
                        .to_owned(),
                );
            }
            config = config.with_optional_profile_coverage_target(
                OptionalProfileCoverageTarget::new(standard_families, challenge_families)
                    .map_err(|error| error.to_string())?,
            );
        }
        _ => {
            return Err(
                "provide both --min-optional-profile-standard-families and --min-optional-profile-challenge-families"
                    .to_owned(),
            );
        }
    }
    let oracle = PinnedOracle::new(OracleConfig {
        java_program: PathBuf::from("java"),
        jar_path: jar,
        policy_packs_path: policy_packs,
    })
    .map_err(|error| error.to_string())?;
    let seeds = source
        .load_sampled_with_profile(config.max_seed_schemas, config.seed, source_profile)
        .map_err(|error| error.to_string())?;
    let (mut dataset, report) = TrainingDataGenerator::new(oracle)
        .generate(&config, &workspace, seeds)
        .map_err(|error| error.to_string())?;
    let generalization = dataset.stamp_benchmark_readiness(config.promotion_manifest.as_ref());
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    dataset.save(&output).map_err(|error| error.to_string())?;
    println!(
        "Generated {} records ({} safe, {} warning, {} breaking, {} rejected, {} duplicates, {} quota-skipped) from {} sampled source schemas: {}",
        dataset.len(),
        report.oracle_safe,
        report.oracle_warning,
        report.oracle_breaking,
        report.oracle_rejected,
        report.duplicates_rejected,
        report.outcome_quota_skipped,
        report.seeds_seen,
        output.display()
    );
    if let Some(reached) = report.coverage_target_reached {
        println!(
            "Coverage-driven result: target_reached={reached}, reasons={:?}",
            report.coverage_reasons
        );
    }
    let readiness = dataset.training_readiness(
        TargetMode::BinaryBreaking,
        DatasetSplitConfig::new(0.70, 0.15, 0.15, seed).map_err(|error| error.to_string())?,
    );
    println!(
        "Dataset report: families={}, policies={}, mutations={}, binary-ready={}, reasons={:?}",
        readiness.statistics.independent_families,
        readiness.statistics.policy_packs,
        readiness.statistics.mutation_categories,
        readiness.ready_for_training,
        readiness.reasons,
    );
    println!(
        "Generalization report: benchmark-ready={}, challenge_families={}, complete_three_policy_pairs={}, shortcut_risks={:?}, reasons={:?}",
        generalization.ready_for_generalization_benchmark,
        generalization.challenge_families,
        generalization.complete_policy_counterfactual_pairs,
        generalization.shortcut_risks,
        generalization.reasons,
    );
    Ok(())
}

fn train(arguments: Vec<String>) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut epochs = 100usize;
    let mut batch_size = 16usize;
    let mut learning_rate = 0.05f64;
    let mut seed = 42u64;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--epochs" => {
                epochs = value
                    .parse()
                    .map_err(|_| format!("--epochs must be a whole number, got {value}"))?
            }
            "--batch-size" => {
                batch_size = value
                    .parse()
                    .map_err(|_| format!("--batch-size must be a whole number, got {value}"))?
            }
            "--learning-rate" => {
                learning_rate = value
                    .parse()
                    .map_err(|_| format!("--learning-rate must be a finite number, got {value}"))?
            }
            "--seed" => {
                seed = value
                    .parse()
                    .map_err(|_| format!("--seed must be a whole number, got {value}"))?
            }
            _ => return Err(format!("Unknown train option: {flag}")),
        }
        index += 2;
    }
    let input = input.ok_or("Missing --input")?;
    let output = output.ok_or("Missing --output")?;
    let dataset = PreparedDcgDataset::load(&input).map_err(|error| error.to_string())?;
    let split =
        DatasetSplitConfig::new(0.70, 0.15, 0.15, seed).map_err(|error| error.to_string())?;
    let model = ModelConfig::new(dataset.records()[0].features.len(), vec![12, 6], 0.5)
        .map_err(|error| error.to_string())?;
    let training = TrainingConfig::new(epochs, batch_size).map_err(|error| error.to_string())?;
    let pipeline = DcgPipelineConfig::new(
        PredictionKind::BreakingChange,
        split,
        model,
        training,
        vec![0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    )
    .map_err(|error| error.to_string())?;
    let result = pipeline
        .run(
            &dataset,
            Sgd::new(learning_rate).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())?;
    let metadata = TrainingMetadata {
        optimizer: "sgd".to_owned(),
        epochs,
        batch_size,
        learning_rate: Some(learning_rate.to_string()),
        seed,
        training_samples: result.split.train().len(),
        validation_samples: result.split.validation().len(),
        test_samples: result.split.test().len(),
        dataset_id: dataset.dataset_version().to_owned(),
    }
    .validate()
    .map_err(|error| error.to_string())?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    ModelArtifact::new_with_feature_version(
        "dcg-oracle-breaking-v2",
        dataset.feature_version(),
        result.model,
        result.scaler,
        metadata,
    )
    .map_err(|error| error.to_string())?
    .save(&output)
    .map_err(|error| error.to_string())?;
    ModelArtifact::load(&output).map_err(|error| error.to_string())?;
    println!(
        "Training complete: train={}, validation={}, test={}, final={:?}, reports={:?}, artifact={}",
        result.split.train().len(),
        result.split.validation().len(),
        result.split.test().len(),
        result.evaluation,
        result.classification_report,
        output.display()
    );
    println!(
        "Loss: train_first={:?}, train_last={:?}, validation_first={:?}, validation_last={:?}",
        result.training_history.epoch_losses().first(),
        result.training_history.epoch_losses().last(),
        result.training_history.validation_losses().first(),
        result.training_history.validation_losses().last(),
    );
    Ok(())
}

/// Runs the normal and three challenge evaluation protocols on one audited
/// prepared dataset. Challenge families never enter a protocol's training,
/// validation, or normal test split.
fn evaluate_protocols(arguments: Vec<String>) -> Result<(), String> {
    let mut input = None;
    let mut epochs = 100usize;
    let mut batch_size = 16usize;
    let mut learning_rate = 0.05f64;
    let mut seeds = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--epochs" => {
                epochs = value
                    .parse()
                    .map_err(|_| format!("--epochs must be a whole number, got {value}"))?
            }
            "--batch-size" => {
                batch_size = value
                    .parse()
                    .map_err(|_| format!("--batch-size must be a whole number, got {value}"))?
            }
            "--learning-rate" => {
                learning_rate = value
                    .parse()
                    .map_err(|_| format!("--learning-rate must be a finite number, got {value}"))?
            }
            "--seed" => seeds.push(
                value
                    .parse()
                    .map_err(|_| format!("--seed must be a whole number, got {value}"))?,
            ),
            "--seeds" => {
                let parsed = value
                    .split(',')
                    .map(str::trim)
                    .filter(|seed| !seed.is_empty())
                    .map(|seed| {
                        seed.parse::<u64>().map_err(|_| {
                            format!("--seeds contains an invalid whole number: {seed}")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if parsed.is_empty() {
                    return Err("--seeds must contain at least one whole number".to_owned());
                }
                seeds.extend(parsed);
            }
            _ => return Err(format!("Unknown evaluate-protocols option: {flag}")),
        }
        index += 2;
    }
    let dataset = PreparedDcgDataset::load(&input.ok_or("Missing --input")?)
        .map_err(|error| error.to_string())?;
    if seeds.is_empty() {
        seeds.push(42);
    }
    seeds.sort_unstable();
    seeds.dedup();
    for seed in seeds {
        println!("evaluation split seed={seed}");
        evaluate_protocols_for_seed(&dataset, epochs, batch_size, learning_rate, seed)?;
    }
    Ok(())
}

fn evaluate_protocols_for_seed(
    dataset: &PreparedDcgDataset,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    seed: u64,
) -> Result<(), String> {
    let standard = standard_protocol_training_set(dataset, |_| true)?;
    let normal = run_binary_protocol(&standard, epochs, batch_size, learning_rate, seed)?;
    println!("protocol normal: test={:?}", normal.evaluation);

    let policies = standard
        .records()
        .iter()
        .map(|record| record.policy_pack.clone())
        .collect::<BTreeSet<_>>();
    for policy in policies {
        let training =
            standard_protocol_training_set(dataset, |record| record.policy_pack != policy)?;
        let challenge = dataset
            .challenge_subset(&ChallengeProtocol::HeldOutPolicy {
                policy_pack: policy.clone(),
            })
            .map_err(|error| error.to_string())?;
        let result = run_binary_protocol(&training, epochs, batch_size, learning_rate, seed)?;
        println!(
            "protocol held-out-policy:{policy}: training_records={}, challenge_records={}, metrics={:?}",
            training.len(),
            challenge.len(),
            evaluate_binary_challenge(&result, &challenge)?
        );
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
        let training = standard_protocol_training_set(dataset, |record| {
            record
                .generation
                .as_ref()
                .is_none_or(|generation| generation.declared_mutation != mutation)
        })?;
        let challenge = dataset
            .challenge_subset(&ChallengeProtocol::HeldOutMutation {
                mutation: mutation.clone(),
            })
            .map_err(|error| error.to_string())?;
        let labels = challenge
            .records()
            .iter()
            .filter_map(|record| {
                record
                    .compatibility_label
                    .map(|label| label.as_str().to_owned())
            })
            .collect::<BTreeSet<_>>();
        if labels.len() < 2 {
            println!(
                "protocol held-out-mutation:{mutation}: unscored traceability challenge; labels={labels:?}"
            );
            continue;
        }
        let result = run_binary_protocol(&training, epochs, batch_size, learning_rate, seed)?;
        println!(
            "protocol held-out-mutation:{mutation}: training_records={}, challenge_records={}, metrics={:?}",
            training.len(),
            challenge.len(),
            evaluate_binary_challenge(&result, &challenge)?
        );
    }

    let paired_challenge = dataset
        .challenge_subset(&ChallengeProtocol::SameMutationDifferentPolicy)
        .map_err(|error| error.to_string())?;
    println!(
        "protocol same-mutation-different-policy: training_records={}, challenge_records={}, metrics={:?}",
        standard.len(),
        paired_challenge.len(),
        evaluate_binary_challenge(&normal, &paired_challenge)?
    );
    Ok(())
}

fn standard_protocol_training_set(
    dataset: &PreparedDcgDataset,
    keep: impl Fn(&dcgaimodel::models::PreparedDcgRecord) -> bool,
) -> Result<PreparedDcgDataset, String> {
    PreparedDcgDataset::new(
        format!("{}-protocol-training", dataset.dataset_version()),
        dataset
            .records()
            .iter()
            .filter(|record| record.dataset_role == DatasetRole::Standard && keep(record))
            .cloned()
            .collect(),
    )
    .map_err(|error| error.to_string())
}

fn run_binary_protocol(
    dataset: &PreparedDcgDataset,
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    seed: u64,
) -> Result<DcgPipelineResult, String> {
    let split =
        DatasetSplitConfig::new(0.70, 0.15, 0.15, seed).map_err(|error| error.to_string())?;
    let model = ModelConfig::new(dataset.records()[0].features.len(), vec![12, 6], 0.5)
        .map_err(|error| error.to_string())?;
    let training = TrainingConfig::new(epochs, batch_size).map_err(|error| error.to_string())?;
    DcgPipelineConfig::new(
        PredictionKind::BreakingChange,
        split,
        model,
        training,
        vec![0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    )
    .map_err(|error| error.to_string())?
    .run(
        dataset,
        Sgd::new(learning_rate).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn evaluate_binary_challenge(
    result: &DcgPipelineResult,
    challenge: &PreparedDcgDataset,
) -> Result<ClassificationMetrics, String> {
    let raw = challenge
        .to_dataset(PredictionKind::BreakingChange)
        .map_err(|error| error.to_string())?;
    let normalized = result
        .scaler
        .transform_dataset(&raw)
        .map_err(|error| error.to_string())?;
    let predictions = result
        .model
        .predict_batch(normalized.features())
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|prediction| {
            prediction
                .label()
                .map(f64::from)
                .ok_or("binary protocol produced a non-classification prediction".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let targets = normalized
        .targets()
        .iter()
        .map(|target| {
            (target.len() == 1)
                .then_some(target[0])
                .ok_or_else(|| "binary protocol target must contain one value".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    evaluate_binary_classification(&predictions, &targets).map_err(|error| error.to_string())
}

fn train_three_way(arguments: Vec<String>) -> Result<(), String> {
    let mut input = None;
    let mut epochs = 100usize;
    let mut batch_size = 16usize;
    let mut learning_rate = 0.05f64;
    let mut seed = 42u64;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--epochs" => {
                epochs = value
                    .parse()
                    .map_err(|_| format!("--epochs must be a whole number, got {value}"))?
            }
            "--batch-size" => {
                batch_size = value
                    .parse()
                    .map_err(|_| format!("--batch-size must be a whole number, got {value}"))?
            }
            "--learning-rate" => {
                learning_rate = value
                    .parse()
                    .map_err(|_| format!("--learning-rate must be a finite number, got {value}"))?
            }
            "--seed" => {
                seed = value
                    .parse()
                    .map_err(|_| format!("--seed must be a whole number, got {value}"))?
            }
            _ => return Err(format!("Unknown train-three-way option: {flag}")),
        }
        index += 2;
    }
    let input = input.ok_or("Missing --input")?;
    let dataset = PreparedDcgDataset::load(&input).map_err(|error| error.to_string())?;
    let readiness = dataset.training_readiness(
        TargetMode::ThreeWayCompatibility,
        DatasetSplitConfig::new(0.70, 0.15, 0.15, seed).map_err(|error| error.to_string())?,
    );
    if !readiness.ready_for_training {
        return Err(format!(
            "Three-way dataset is not structurally ready: {:?}",
            readiness.reasons
        ));
    }
    let result = run_three_way_compatibility_pipeline(
        &dataset,
        DatasetSplitConfig::new(0.70, 0.15, 0.15, seed).map_err(|error| error.to_string())?,
        ModelConfig::new(dataset.records()[0].features.len(), vec![12, 6], 0.5)
            .map_err(|error| error.to_string())?,
        TrainingConfig::new(epochs, batch_size).map_err(|error| error.to_string())?,
        Sgd::new(learning_rate).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    println!(
        "Three-way training complete: train={}, validation={}, test={}, evaluation={:?}",
        result.split.train().len(),
        result.split.validation().len(),
        result.split.test().len(),
        result.evaluation,
    );
    println!(
        "Cross-entropy: train_first={:?}, train_last={:?}, validation_first={:?}, validation_last={:?}",
        result.training_history.epoch_losses().first(),
        result.training_history.epoch_losses().last(),
        result.training_history.validation_losses().first(),
        result.training_history.validation_losses().last(),
    );
    Ok(())
}

fn evaluate_three_way_mutation(arguments: Vec<String>) -> Result<(), String> {
    let mut input = None;
    let mut mutation = None;
    let mut variant = None;
    let mut epochs = 100usize;
    let mut batch_size = 16usize;
    let mut learning_rate = 0.05f64;
    let mut seed = 42u64;
    let mut index = 0;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        match flag.as_str() {
            "--input" => input = Some(PathBuf::from(value)),
            "--mutation" => mutation = Some(value.clone()),
            "--variant" => variant = Some(value.clone()),
            "--epochs" => {
                epochs = value
                    .parse()
                    .map_err(|_| format!("--epochs must be a whole number, got {value}"))?
            }
            "--batch-size" => {
                batch_size = value
                    .parse()
                    .map_err(|_| format!("--batch-size must be a whole number, got {value}"))?
            }
            "--learning-rate" => {
                learning_rate = value
                    .parse()
                    .map_err(|_| format!("--learning-rate must be a finite number, got {value}"))?
            }
            "--seed" => {
                seed = value
                    .parse()
                    .map_err(|_| format!("--seed must be a whole number, got {value}"))?
            }
            _ => {
                return Err(format!(
                    "Unknown evaluate-three-way-mutation option: {flag}"
                ));
            }
        }
        index += 2;
    }
    let dataset = PreparedDcgDataset::load(&input.ok_or("Missing --input")?)
        .map_err(|error| error.to_string())?;
    let mutation = mutation.ok_or("Missing --mutation")?;
    let held_out_variant = variant.clone();
    let training = standard_protocol_training_set(&dataset, |record| {
        record.generation.as_ref().is_none_or(|generation| {
            generation.declared_mutation != mutation
                || held_out_variant.as_ref().is_some_and(|variant| {
                    structural_variant_key(generation).as_deref() != Some(variant)
                })
        })
    })?;
    let challenge = if let Some(variant) = &variant {
        PreparedDcgDataset::new(
            format!(
                "{}-challenge-{mutation}-{variant}",
                dataset.dataset_version()
            ),
            dataset
                .records()
                .iter()
                .filter(|record| {
                    record.dataset_role == DatasetRole::Challenge
                        && record.generation.as_ref().is_some_and(|generation| {
                            generation.declared_mutation == mutation
                                && structural_variant_key(generation).as_deref()
                                    == Some(variant.as_str())
                        })
                })
                .cloned()
                .collect(),
        )
        .map_err(|error| error.to_string())?
    } else {
        dataset
            .challenge_subset(&ChallengeProtocol::HeldOutMutation {
                mutation: mutation.clone(),
            })
            .map_err(|error| error.to_string())?
    };
    let labels = challenge
        .records()
        .iter()
        .filter_map(|record| {
            record
                .compatibility_label
                .map(|label| label.as_str().to_owned())
        })
        .collect::<BTreeSet<_>>();
    if labels.len() < 2 {
        return Err(format!(
            "held-out mutation `{mutation}` is traceability-only with labels={labels:?}"
        ));
    }
    let result = run_three_way_compatibility_pipeline(
        &training,
        DatasetSplitConfig::new(0.70, 0.15, 0.15, seed).map_err(|error| error.to_string())?,
        ModelConfig::new(training.records()[0].features.len(), vec![12, 6], 0.5)
            .map_err(|error| error.to_string())?,
        TrainingConfig::new(epochs, batch_size).map_err(|error| error.to_string())?,
        Sgd::new(learning_rate).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let challenge = challenge
        .to_target_dataset(TargetMode::ThreeWayCompatibility)
        .map_err(|error| error.to_string())?;
    let challenge = result
        .scaler
        .transform_dataset(&challenge)
        .map_err(|error| error.to_string())?;
    let predictions = challenge
        .features()
        .iter()
        .map(|features| result.model.predict(features))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| error.to_string())?;
    let evaluation =
        evaluate_three_way(&predictions, challenge.targets()).map_err(|error| error.to_string())?;
    println!(
        "three-way held-out-mutation:{mutation}: variant={variant:?}, training_records={}, challenge_records={}, labels={labels:?}, metrics={evaluation:?}",
        training.len(),
        challenge.len()
    );
    Ok(())
}
