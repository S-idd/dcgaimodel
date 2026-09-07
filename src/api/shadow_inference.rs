use crate::features::{
    ApprovedPolicyContexts, DCG_FEATURE_V6_VERSION, SchemaChangeFeatureV6Extractor,
};
use crate::models::{CompatibilityLabel, ThreeWayModelArtifact};
use axum::Router;
use axum::extract::{Json, State, rejection::JsonRejection};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::net::TcpListener;

/// The only policy document accepted by the frozen V9 shadow service.
pub const FROZEN_V9_POLICY_PACK_SHA256: &str =
    "8f82b058f81ace43c89180803c7ec26ac734b84d0092036a77115688337e1bb6";
const V9_DATASET_SHA256: &str = "fa09e645480737ba856940778264284d8e832f4532e989c92a259320cfe05ad3";
const V9_TRAINING_ORACLE_SHA256: &str =
    "809b25e627e43f847f00a0ce87dcde33ad359006bf22be403e26d662274677dc";
const POLICY_RELATIVE_PATH: &str = "data/inference/frozen-v9/policy-packs-v5-compositional.json";

#[derive(Debug, Clone, Copy)]
struct FrozenModelSpec {
    seed: &'static str,
    relative_path: &'static str,
    sha256: &'static str,
}

const FROZEN_MODELS: [FrozenModelSpec; 3] = [
    FrozenModelSpec {
        seed: "20260826",
        relative_path: "data/experiments/v9-multiclass-cpu-100e/models/seed-20260826-normal-family-split.json",
        sha256: "5da2fedbee5d1b3c84c79cb75e2cd10c0b3462066b66571570e93fb7ccd84988",
    },
    FrozenModelSpec {
        seed: "20260827",
        relative_path: "data/experiments/v9-multiclass-cpu-100e/models/seed-20260827-normal-family-split.json",
        sha256: "bd464322c272b8ec1d5ab88605022d4a48e3738b63ab1791a4c7e56f37222ac8",
    },
    FrozenModelSpec {
        seed: "20260828",
        relative_path: "data/experiments/v9-multiclass-cpu-100e/models/seed-20260828-normal-family-split.json",
        sha256: "24ec9e4e758370093c228af34e3b05ac65820b3fd413ca80a269206ce1edd2c0",
    },
];

/// Raw schema transition accepted by `POST /v1/shadow/predict`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub base_schema: Value,
    pub candidate_schema: Value,
    pub policy_pack: String,
}

/// Named SAFE/WARNING/BREAKING probabilities from one seed model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbabilityResponse {
    pub safe: f64,
    pub warning: f64,
    pub breaking: f64,
}

/// Raw prediction from one frozen V9 seed model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SeedPredictionResponse {
    pub seed: String,
    pub label: String,
    pub probabilities: ProbabilityResponse,
}

/// The v1 response deliberately contains raw seed outputs and no aggregation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowInferenceResponse {
    pub predictions: Vec<SeedPredictionResponse>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    body: ErrorResponse,
}

impl ApiError {
    fn bad_request(code: &str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorResponse {
                error: ErrorBody {
                    code: code.to_owned(),
                    message: message.into(),
                },
            },
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ErrorResponse {
                error: ErrorBody {
                    code: "INFERENCE_FAILED".to_owned(),
                    message: message.into(),
                },
            },
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: ErrorResponse {
                error: ErrorBody {
                    code: "NOT_READY".to_owned(),
                    message: message.into(),
                },
            },
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

#[derive(Debug)]
struct FrozenSeedModel {
    seed: &'static str,
    artifact: ThreeWayModelArtifact,
}

/// Immutable, inference-only state loaded from the three frozen V9 artifacts.
#[derive(Debug)]
pub struct ShadowInferenceService {
    policy_contexts: ApprovedPolicyContexts,
    declared_policy_packs: BTreeSet<String>,
    models: Vec<FrozenSeedModel>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadinessResponse {
    pub status: String,
    pub service: String,
    pub feature_version: String,
    pub policy_sha256: String,
    pub model_seeds: Vec<String>,
}

impl ShadowInferenceService {
    /// Loads the pinned policy snapshot and all three frozen V9 models read-only.
    pub fn load_frozen_v9(project_root: impl AsRef<Path>) -> Result<Self, String> {
        let project_root = project_root.as_ref();
        let policy_path = project_root.join(POLICY_RELATIVE_PATH);
        let policy_bytes = read_verified(&policy_path, FROZEN_V9_POLICY_PACK_SHA256)?;
        let policy_document = std::str::from_utf8(&policy_bytes)
            .map_err(|error| format!("pinned policy file is not UTF-8: {error}"))?;
        let policy_json: Value = serde_json::from_str(policy_document)
            .map_err(|error| format!("pinned policy file is invalid JSON: {error}"))?;
        let declared_policy_packs = policy_json
            .get("packs")
            .and_then(Value::as_object)
            .ok_or_else(|| "pinned policy file has no object-valued `packs` field".to_owned())?
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        if declared_policy_packs.is_empty() {
            return Err("pinned policy file declares no policy packs".to_owned());
        }
        let policy_names = declared_policy_packs.iter().cloned().collect::<Vec<_>>();
        let policy_contexts = ApprovedPolicyContexts::from_json(policy_document, &policy_names)
            .map_err(|error| format!("could not resolve pinned policy contexts: {error}"))?;

        let mut models = Vec::with_capacity(FROZEN_MODELS.len());
        for spec in FROZEN_MODELS {
            let model_path = project_root.join(spec.relative_path);
            let model_bytes = read_verified(&model_path, spec.sha256)?;
            let artifact = ThreeWayModelArtifact::from_slice(&model_bytes).map_err(|error| {
                format!("could not load frozen seed {} model: {error}", spec.seed)
            })?;
            verify_artifact_identity(spec, &artifact)?;
            models.push(FrozenSeedModel {
                seed: spec.seed,
                artifact,
            });
        }

        Ok(Self {
            policy_contexts,
            declared_policy_packs,
            models,
        })
    }

    fn predict(&self, request: &InferenceRequest) -> Result<ShadowInferenceResponse, ApiError> {
        if !self.declared_policy_packs.contains(&request.policy_pack) {
            return Err(ApiError::bad_request(
                "UNKNOWN_POLICY_PACK",
                format!(
                    "policy pack `{}` is not declared by the pinned policy file",
                    request.policy_pack
                ),
            ));
        }
        validate_supported_schema("base_schema", &request.base_schema)?;
        validate_supported_schema("candidate_schema", &request.candidate_schema)?;

        let policy_context = self
            .policy_contexts
            .get(&request.policy_pack)
            .map_err(|error| ApiError::bad_request("UNKNOWN_POLICY_PACK", error.to_string()))?;
        let base_schema = serde_json::to_string(&request.base_schema).map_err(|error| {
            ApiError::internal(format!("could not encode base schema: {error}"))
        })?;
        let candidate_schema =
            serde_json::to_string(&request.candidate_schema).map_err(|error| {
                ApiError::internal(format!("could not encode candidate schema: {error}"))
            })?;
        let features = SchemaChangeFeatureV6Extractor::new()
            .extract(&base_schema, &candidate_schema, policy_context)
            .map_err(|error| {
                ApiError::bad_request("SCHEMA_FEATURE_EXTRACTION_FAILED", error.to_string())
            })?;

        let mut predictions = Vec::with_capacity(self.models.len());
        for seed_model in &self.models {
            let normalized = seed_model
                .artifact
                .scaler()
                .transform_vector(&features)
                .map_err(|error| ApiError::internal(format!("frozen scaler failed: {error}")))?;
            let prediction = seed_model
                .artifact
                .model()
                .predict(&normalized)
                .map_err(|error| ApiError::internal(format!("frozen model failed: {error}")))?;
            let probabilities = prediction.probabilities.iter().copied().collect::<Vec<_>>();
            let [safe, warning, breaking] = probabilities.as_slice() else {
                return Err(ApiError::internal(
                    "frozen model returned a non-three-way probability vector",
                ));
            };
            predictions.push(SeedPredictionResponse {
                seed: seed_model.seed.to_owned(),
                label: uppercase_label(prediction.label).to_owned(),
                probabilities: ProbabilityResponse {
                    safe: *safe,
                    warning: *warning,
                    breaking: *breaking,
                },
            });
        }
        Ok(ShadowInferenceResponse { predictions })
    }

    fn readiness(&self) -> Result<ReadinessResponse, ApiError> {
        let policy_pack = self
            .declared_policy_packs
            .first()
            .ok_or_else(|| ApiError::unavailable("no frozen policy packs are loaded"))?;
        self.predict(&InferenceRequest {
            base_schema: serde_json::json!({"type": "object"}),
            candidate_schema: serde_json::json!({"type": "object"}),
            policy_pack: policy_pack.clone(),
        })
        .map_err(|error| {
            ApiError::unavailable(format!(
                "readiness inference failed: {}",
                error.body.error.message
            ))
        })?;

        Ok(ReadinessResponse {
            status: "UP".to_owned(),
            service: "dcgaimodel-shadow-inference".to_owned(),
            feature_version: DCG_FEATURE_V6_VERSION.to_owned(),
            policy_sha256: FROZEN_V9_POLICY_PACK_SHA256.to_owned(),
            model_seeds: self
                .models
                .iter()
                .map(|model| model.seed.to_owned())
                .collect(),
        })
    }
}

/// Constructs the HTTP router without binding a socket, enabling in-process tests.
pub fn frozen_v9_router(service: ShadowInferenceService) -> Router {
    Router::new()
        .route("/health/ready", axum::routing::get(readiness_handler))
        .route("/v1/shadow/predict", post(predict_handler))
        .with_state(Arc::new(service))
}

/// Serves the frozen V9 shadow endpoint on an already-bound listener.
pub async fn serve_frozen_v9(
    listener: TcpListener,
    service: ShadowInferenceService,
) -> std::io::Result<()> {
    axum::serve(listener, frozen_v9_router(service)).await
}

async fn predict_handler(
    State(service): State<Arc<ShadowInferenceService>>,
    payload: Result<Json<InferenceRequest>, JsonRejection>,
) -> Result<Json<ShadowInferenceResponse>, ApiError> {
    let Json(request) = payload.map_err(|error| {
        ApiError::bad_request(
            "INVALID_REQUEST_JSON",
            format!("request body does not match the inference contract: {error}"),
        )
    })?;
    service.predict(&request).map(Json)
}

async fn readiness_handler(
    State(service): State<Arc<ShadowInferenceService>>,
) -> Result<Json<ReadinessResponse>, ApiError> {
    service.readiness().map(Json)
}

fn verify_artifact_identity(
    spec: FrozenModelSpec,
    artifact: &ThreeWayModelArtifact,
) -> Result<(), String> {
    if artifact.feature_version() != DCG_FEATURE_V6_VERSION {
        return Err(format!(
            "frozen seed {} model uses feature version {}, expected {DCG_FEATURE_V6_VERSION}",
            spec.seed,
            artifact.feature_version()
        ));
    }
    if artifact.training_metadata().seed.to_string() != spec.seed {
        return Err(format!(
            "frozen seed {} model records training seed {}",
            spec.seed,
            artifact.training_metadata().seed
        ));
    }
    let provenance = artifact.input_provenance();
    if provenance.dataset_sha256 != V9_DATASET_SHA256
        || provenance.oracle_jar_sha256 != V9_TRAINING_ORACLE_SHA256
        || provenance.policy_packs_sha256 != FROZEN_V9_POLICY_PACK_SHA256
    {
        return Err(format!(
            "frozen seed {} model does not carry the registered V9 training provenance",
            spec.seed
        ));
    }
    Ok(())
}

fn read_verified(path: &Path, expected_sha256: &str) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("could not read frozen artifact {}: {error}", path.display()))?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected_sha256 {
        return Err(format!(
            "frozen artifact {} has SHA-256 {actual}, expected {expected_sha256}",
            path.display()
        ));
    }
    Ok(bytes)
}

fn validate_supported_schema(field: &str, schema: &Value) -> Result<(), ApiError> {
    jsonschema::meta::validate(schema).map_err(|error| {
        ApiError::bad_request(
            "INVALID_SCHEMA",
            format!("{field} is not a valid JSON Schema: {error}"),
        )
    })?;
    validate_type_bearing_node(field, "$", schema)
}

fn validate_type_bearing_node(field: &str, path: &str, schema: &Value) -> Result<(), ApiError> {
    let object = schema.as_object().ok_or_else(|| {
        ApiError::bad_request(
            "UNSUPPORTED_SCHEMA",
            format!("{field} at {path} must be an object-valued, type-bearing schema"),
        )
    })?;
    let schema_type = object.get("type").ok_or_else(|| {
        ApiError::bad_request(
            "UNSUPPORTED_SCHEMA",
            format!(
                "{field} at {path} has no `type`; type-less and $ref-only nodes are outside the V6 inference contract"
            ),
        )
    })?;
    validate_type_keyword(field, path, schema_type)?;

    if let Some(additional) = object.get("additionalProperties")
        && !additional.is_boolean()
    {
        return Err(ApiError::bad_request(
            "UNSUPPORTED_SCHEMA",
            format!(
                "{field} at {path} uses schema-valued `additionalProperties`, which V6 does not extract"
            ),
        ));
    }
    if let Some(properties) = object.get("properties") {
        let properties = properties.as_object().ok_or_else(|| {
            ApiError::bad_request(
                "INVALID_SCHEMA",
                format!("{field} at {path}.properties must be an object"),
            )
        })?;
        for (name, child) in properties {
            validate_type_bearing_node(field, &format!("{path}.properties.{name}"), child)?;
        }
    }
    if let Some(items) = object.get("items") {
        validate_type_bearing_node(field, &format!("{path}.items"), items)?;
    }
    Ok(())
}

fn validate_type_keyword(field: &str, path: &str, value: &Value) -> Result<(), ApiError> {
    const TYPES: [&str; 7] = [
        "null", "boolean", "object", "array", "number", "string", "integer",
    ];
    let valid = match value {
        Value::String(value) => TYPES.contains(&value.as_str()),
        Value::Array(values) => {
            !values.is_empty()
                && values
                    .iter()
                    .all(|value| value.as_str().is_some_and(|value| TYPES.contains(&value)))
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(ApiError::bad_request(
            "INVALID_SCHEMA",
            format!("{field} at {path} has an invalid `type` keyword"),
        ))
    }
}

const fn uppercase_label(label: CompatibilityLabel) -> &'static str {
    match label {
        CompatibilityLabel::Safe => "SAFE",
        CompatibilityLabel::Warning => "WARNING",
        CompatibilityLabel::Breaking => "BREAKING",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, header};
    use std::path::PathBuf;
    use tower::ServiceExt;

    fn project_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn app() -> Router {
        frozen_v9_router(ShadowInferenceService::load_frozen_v9(project_root()).unwrap())
    }

    fn schema_with(properties: Value) -> Value {
        serde_json::json!({"type": "object", "properties": properties})
    }

    async fn post_json(body: Value) -> Response {
        app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/shadow/predict")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn get_readiness() -> Response {
        app()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/health/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn response_json<T: serde::de::DeserializeOwned>(response: Response) -> T {
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
    }

    #[tokio::test]
    async fn valid_request_returns_three_raw_seed_predictions() {
        let response = post_json(serde_json::json!({
            "base_schema": schema_with(serde_json::json!({"id": {"type": "string"}})),
            "candidate_schema": schema_with(serde_json::json!({
                "id": {"type": "string"},
                "status": {"type": "string", "enum": ["active"]}
            })),
            "policy_pack": "baseline"
        }))
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: ShadowInferenceResponse = response_json(response).await;
        assert_eq!(
            body.predictions
                .iter()
                .map(|prediction| prediction.seed.as_str())
                .collect::<Vec<_>>(),
            ["20260826", "20260827", "20260828"]
        );
        assert!(body.predictions.iter().all(|prediction| {
            matches!(prediction.label.as_str(), "SAFE" | "WARNING" | "BREAKING")
                && (prediction.probabilities.safe
                    + prediction.probabilities.warning
                    + prediction.probabilities.breaking
                    - 1.0)
                    .abs()
                    < 1e-12
        }));
    }

    #[tokio::test]
    async fn readiness_reports_frozen_identity_after_valid_inference() {
        let response = get_readiness().await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: ReadinessResponse = response_json(response).await;
        assert_eq!(body.status, "UP");
        assert_eq!(body.service, "dcgaimodel-shadow-inference");
        assert_eq!(body.feature_version, "dcg-features-v6");
        assert_eq!(body.policy_sha256, FROZEN_V9_POLICY_PACK_SHA256);
        assert_eq!(body.model_seeds, ["20260826", "20260827", "20260828"]);
    }

    #[tokio::test]
    async fn unknown_policy_pack_is_a_clear_bad_request() {
        let response = post_json(serde_json::json!({
            "base_schema": schema_with(serde_json::json!({"id": {"type": "string"}})),
            "candidate_schema": schema_with(serde_json::json!({"id": {"type": "string"}})),
            "policy_pack": "does-not-exist"
        }))
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: ErrorResponse = response_json(response).await;
        assert_eq!(body.error.code, "UNKNOWN_POLICY_PACK");
        assert!(body.error.message.contains("does-not-exist"));
    }

    #[tokio::test]
    async fn malformed_request_json_is_a_clear_bad_request() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/shadow/predict")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(r#"{"base_schema": "#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: ErrorResponse = response_json(response).await;
        assert_eq!(body.error.code, "INVALID_REQUEST_JSON");
    }

    #[tokio::test]
    async fn unsupported_type_less_schema_is_a_clear_bad_request() {
        let response = post_json(serde_json::json!({
            "base_schema": {
                "type": "object",
                "properties": {"item": {"$ref": "#/definitions/Item"}}
            },
            "candidate_schema": schema_with(serde_json::json!({"item": {"type": "object"}})),
            "policy_pack": "baseline"
        }))
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: ErrorResponse = response_json(response).await;
        assert_eq!(body.error.code, "UNSUPPORTED_SCHEMA");
        assert!(body.error.message.contains("$ref-only"));
    }

    #[tokio::test]
    async fn invalid_schema_keyword_is_a_clear_bad_request() {
        let response = post_json(serde_json::json!({
            "base_schema": {"type": 42},
            "candidate_schema": schema_with(serde_json::json!({"id": {"type": "string"}})),
            "policy_pack": "baseline"
        }))
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: ErrorResponse = response_json(response).await;
        assert_eq!(body.error.code, "INVALID_SCHEMA");
        assert!(body.error.message.contains("base_schema"));
    }

    #[tokio::test]
    async fn http_facade_matches_persisted_external_evaluation_predictions() {
        let manifest: Value = serde_json::from_slice(
            &fs::read(project_root().join("data/external/kubernetes-openapi-v1/scored-accepted-v1/kubernetes-accepted-external-manifest-v3.json")).unwrap(),
        )
        .unwrap();
        let transition = manifest["transitions"]
            .as_array()
            .unwrap()
            .iter()
            .find(|transition| {
                transition["record_id"]
                    == "kubernetes.prescreen.v1.21.0-to-v1.22.0.io.k8s.api.networking.v1.HTTPIngressPath"
            })
            .unwrap();
        let response = post_json(serde_json::json!({
            "base_schema": transition["base_schema"].clone(),
            "candidate_schema": transition["candidate_schema"].clone(),
            "policy_pack": transition["policy_pack"].clone()
        }))
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: ShadowInferenceResponse = response_json(response).await;

        for prediction in &body.predictions {
            let report_path = project_root().join(format!(
                "data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/seed-{}-report-v3.json",
                prediction.seed
            ));
            let report: Value = serde_json::from_slice(&fs::read(report_path).unwrap()).unwrap();
            let expected = report["records"]
                .as_array()
                .unwrap()
                .iter()
                .find(|record| record["record_id"] == transition["record_id"])
                .unwrap();
            assert_eq!(prediction.label, "BREAKING");
            assert_eq!(
                [
                    prediction.probabilities.safe,
                    prediction.probabilities.warning,
                    prediction.probabilities.breaking,
                ],
                [
                    expected["predicted_probabilities"][0].as_f64().unwrap(),
                    expected["predicted_probabilities"][1].as_f64().unwrap(),
                    expected["predicted_probabilities"][2].as_f64().unwrap(),
                ]
            );
        }
    }
}
