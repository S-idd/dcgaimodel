//! Deterministic feature engineering for data-contract governance (DCG).
//!
//! This module deliberately knows about contract changes but not neural-network
//! internals, optimizers, or training loops.

mod contract;
mod extractor;

pub use contract::{
    CompatibilityStatus, ContractChange, ContractFeatures, ContractMetadata, SemanticVersion,
};
pub use extractor::{ContractFeatureExtractor, FeatureError, FeatureExtractor};
