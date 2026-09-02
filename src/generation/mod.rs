//! Offline DCG training-data generation primitives.
//!
//! The executable oracle is the sole authority for generated labels. This
//! module stages portable contract files, invokes the pinned JAR, and exposes
//! its result without reimplementing compatibility policy in Rust.

mod conformance;
mod dataset;
mod equivalence;
mod mutation;
mod oracle;

pub use conformance::{
    build_forward_removal_mechanism_audit, build_matched_optional_field_conformance_matrix,
    build_oracle_conformance_matrix,
};
pub use dataset::{
    CoverageTarget, GeneratorConfig, GeneratorReport, JsonSchemaBenchSource,
    OptionalProfileCoverageTarget, RefeatureReport, SeedSchema, SourceSamplingProfile,
    TrainingDataGenerator, canonical_pair_fingerprint, canonical_pair_fingerprint_for_mode,
    refeature_v5_as_v6,
};
pub use equivalence::{
    BACKWARD_V10_JAR_SHA256, BackwardEquivalenceAuditConfig, BackwardEquivalenceAuditReport,
    HISTORICAL_V9_DATASET_SHA256, HISTORICAL_V9_JAR_SHA256, PINNED_POLICY_PACKS_SHA256,
    run_backward_equivalence_audit,
};
pub use mutation::{MutationCandidate, MutationKind, candidates_for_schema};
pub use oracle::{
    GeneratedPair, GeneratorError, OracleConfig, OracleOutcome, OracleRun, PinnedOracle,
    has_fatal_oracle_runtime_failure,
};
