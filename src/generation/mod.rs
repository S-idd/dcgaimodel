//! Offline DCG training-data generation primitives.
//!
//! The executable oracle is the sole authority for generated labels. This
//! module stages portable contract files, invokes the pinned JAR, and exposes
//! its result without reimplementing compatibility policy in Rust.

mod conformance;
mod dataset;
mod mutation;
mod oracle;

pub use conformance::build_oracle_conformance_matrix;
pub use dataset::{
    CoverageTarget, GeneratorConfig, GeneratorReport, JsonSchemaBenchSource,
    OptionalProfileCoverageTarget, RefeatureReport, SeedSchema, SourceSamplingProfile,
    TrainingDataGenerator, canonical_pair_fingerprint, canonical_pair_fingerprint_for_mode,
    refeature_v5_as_v6,
};
pub use mutation::{MutationCandidate, MutationKind, candidates_for_schema};
pub use oracle::{
    GeneratedPair, GeneratorError, OracleConfig, OracleOutcome, OracleRun, PinnedOracle,
    has_fatal_oracle_runtime_failure,
};
