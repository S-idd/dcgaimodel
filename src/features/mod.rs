//! Deterministic feature engineering for data-contract governance (DCG).
//!
//! This module deliberately knows about contract changes but not neural-network
//! internals, optimizers, or training loops.

mod contract;
mod extractor;
mod policy;
mod v2;
mod v3;
mod v4;
mod v5;
mod v6;

pub use contract::{
    CompatibilityStatus, ContractChange, ContractFeatures, ContractMetadata, DCG_FEATURE_COUNT,
    DCG_FEATURE_NAMES, DCG_FEATURE_VERSION, SemanticVersion, feature_schema,
};
mod fixtures;
pub use extractor::{ContractFeatureExtractor, FeatureError, FeatureExtractor};
pub use fixtures::{DcgFixture, FixturePolicyPack, FixtureScenario, realistic_fixtures};
pub use policy::{
    ApprovedPolicyContexts, POLICY_ACTION_COUNT, POLICY_RULE_ACTION_FEATURE_COUNT,
    POLICY_RULE_COUNT, POLICY_RULE_IDS, PolicyFeatureContext,
};
pub use v2::{
    DCG_FEATURE_V2_COUNT, DCG_FEATURE_V2_NAMES, DCG_FEATURE_V2_VERSION,
    SchemaChangeFeatureExtractor, feature_count_for_version, validate_feature_vector_for_version,
    validate_feature_vector_v2,
};
pub use v3::{
    DCG_FEATURE_V3_COUNT, DCG_FEATURE_V3_NAMES, DCG_FEATURE_V3_VERSION,
    SchemaChangeFeatureV3Extractor, validate_feature_vector_v3,
};
pub use v4::{
    DCG_FEATURE_V4_COUNT, DCG_FEATURE_V4_NAMES, DCG_FEATURE_V4_VERSION,
    SchemaChangeFeatureV4Extractor, validate_feature_vector_v4,
};
pub use v5::{
    DCG_FEATURE_V5_COUNT, DCG_FEATURE_V5_NAMES, DCG_FEATURE_V5_VERSION,
    SchemaChangeFeatureV5Extractor, validate_feature_vector_v5,
};
pub use v6::{
    DCG_FEATURE_V6_COUNT, DCG_FEATURE_V6_NAMES, DCG_FEATURE_V6_VERSION,
    SchemaChangeFeatureV6Extractor, validate_feature_vector_v6,
};
