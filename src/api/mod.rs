//! Network boundaries for advisory DCG model inference.

mod shadow_inference;

pub use shadow_inference::{
    ErrorBody, ErrorResponse, FROZEN_V9_POLICY_PACK_SHA256, InferenceRequest, ProbabilityResponse,
    SeedPredictionResponse, ShadowInferenceResponse, ShadowInferenceService, frozen_v9_router,
    serve_frozen_v9,
};
