//! Reusable DCG model wrappers and dataset preparation.

mod artifact;
mod data;
mod experiment;
mod external_evaluation;
mod model;
mod oracle_evidence;
mod pipeline;
mod prepared_dataset;

pub use artifact::{
    ARTIFACT_FORMAT_VERSION, ModelArtifact, ModelArtifactError, THREE_WAY_ARTIFACT_FORMAT_VERSION,
    THREE_WAY_CLASS_ORDER, ThreeWayModelArtifact, TrainingInputProvenance, TrainingMetadata,
};
pub use data::{
    CompatibilityLabel, ContractExample, ContractLabels, DcgDatasetError, DcgTrainingRecord,
    TargetMode, build_dataset, synthetic_examples,
};
pub use experiment::{
    BINARY_THREE_SEED_EXPERIMENT_FORMAT_VERSION, BinaryExperimentMetrics, BinarySeedRunReport,
    BinaryThreeSeedExperimentReport, CHALLENGE_NEAR_DUPLICATE_AUDIT_FORMAT_VERSION,
    ChallengeNearDuplicateAuditReport, ChallengeProtocolNearDuplicateAudit,
    ExperimentConfigurationReport, FeatureOverlapSummary, NearDuplicateSample,
    THREE_WAY_EXPERIMENT_FORMAT_VERSION, ThreeWayExperimentConfig, ThreeWayExperimentReport,
    ThreeWayMetrics, ThreeWayProtocolAggregate, ThreeWayProtocolResult, ThreeWaySeedRunReport,
    audit_challenge_near_duplicates, run_binary_three_seed_experiment, run_three_way_experiment,
    structural_variant_key,
};
pub use external_evaluation::{
    EXTERNAL_EVALUATION_REPORT_FORMAT_VERSION, EXTERNAL_TRANSITION_MANIFEST_FORMAT_VERSION,
    ExternalDualProvenance, ExternalEvaluationConfig, ExternalEvaluationReport,
    ExternalFeatureCoordinate, ExternalFeatureCoordinateDifference,
    ExternalFeatureDiagnosticConfig, ExternalFeatureDiagnosticNeighbor,
    ExternalFeatureDiagnosticReport, ExternalManifestPreflightConfig,
    ExternalManifestPreflightRecord, ExternalManifestPreflightReport, ExternalMetricReport,
    ExternalOracleEvidence, ExternalSourceKind, ExternalTransitionEvaluationRecord,
    ExternalV9Reference, diagnose_external_feature_space, evaluate_external_transitions,
    preflight_external_manifest,
};
pub use model::{
    DcgModel, ModelConfig, ModelError, ThreeWayCompatibilityModel, ThreeWayPrediction,
};
pub use oracle_evidence::{
    FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_FILE_NAME, FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_SHA256,
    InvariantEvidenceConfig, ORACLE_CONFORMANCE_FORMAT_VERSION, ORACLE_PROMOTION_FORMAT_VERSION,
    OracleCompatibilityMode, OracleConformanceCell, OracleConformancePreflight,
    OracleConformanceReport, OracleConformanceRun, OracleCrossPolicyEvidence,
    OracleInvariantPromotionManifest, QualifiedOracleInvariantPromotion,
};
pub use pipeline::{
    ClassificationPipelineReport, DcgPipelineConfig, DcgPipelineError, DcgPipelineResult,
    EvaluationResult, ThreeWayEvaluation, ThreeWayPipelineResult, evaluate_three_way,
    run_three_way_compatibility_pipeline,
};
pub use prepared_dataset::{
    ChallengeProtocol, ChallengeProtocolReport, DatasetRole, DatasetStatistics,
    GeneralizationReadinessReport, GeneratedRecordProvenance, OptionalFieldProfileCoverage,
    OracleInvariantException, PREPARED_DATASET_FORMAT_VERSION, PreparedDatasetError,
    PreparedDcgDataset, PreparedDcgDatasetSplit, PreparedDcgRecord, TrainingReadinessReport,
};
