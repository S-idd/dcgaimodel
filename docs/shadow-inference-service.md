# Rust shadow inference service

This boundary is Rust-only. Spring Boot is not part of this step and must not
extract, order, normalize, or otherwise interpret model features.

## Endpoint

`POST /v1/shadow/predict` accepts JSON:

```json
{
  "base_schema": {
    "type": "object",
    "properties": {
      "id": { "type": "string" }
    }
  },
  "candidate_schema": {
    "type": "object",
    "properties": {
      "id": { "type": "string" },
      "note": { "type": "string" }
    }
  },
  "policy_pack": "baseline"
}
```

A successful response contains the three frozen seed outputs separately:

```json
{
  "predictions": [
    {
      "seed": "20260826",
      "label": "SAFE|WARNING|BREAKING",
      "probabilities": {
        "safe": 0.0,
        "warning": 0.0,
        "breaking": 0.0
      }
    },
    {
      "seed": "20260827",
      "label": "SAFE|WARNING|BREAKING",
      "probabilities": {
        "safe": 0.0,
        "warning": 0.0,
        "breaking": 0.0
      }
    },
    {
      "seed": "20260828",
      "label": "SAFE|WARNING|BREAKING",
      "probabilities": {
        "safe": 0.0,
        "warning": 0.0,
        "breaking": 0.0
      }
    }
  ]
}
```

Version 1 deliberately has no aggregate label or aggregate probability. An
aggregation rule has not been evaluated or approved, so the boundary exposes
only the raw output of each frozen seed.

Client rejections use HTTP 400 and a stable envelope:

```json
{
  "error": {
    "code": "UNKNOWN_POLICY_PACK",
    "message": "policy pack `unknown` is not declared by the pinned policy file"
  }
}
```

The client error codes are:

- `INVALID_REQUEST_JSON`: malformed JSON or a body that does not match the
  request contract;
- `UNKNOWN_POLICY_PACK`: the exact policy name is absent from the pinned file;
- `INVALID_SCHEMA`: the JSON value is not a valid JSON Schema;
- `UNSUPPORTED_SCHEMA`: the schema is valid JSON Schema but is outside the V6
  inference subset, including boolean schemas, type-less/$ref-only structural
  nodes, tuple-valued `items`, and schema-valued `additionalProperties`;
- `SCHEMA_FEATURE_EXTRACTION_FAILED`: the verified V6 extractor rejected the
  otherwise boundary-valid input.

Unexpected scaler/model failures return HTTP 500 with `INFERENCE_FAILED`; they
are responses, not panics.

## Readiness

`GET /health/ready` returns HTTP 200 only after the pinned policy and all three
frozen model artifacts have loaded and a valid inference probe succeeds. The
response reports the feature version, policy SHA-256, and model seeds without
exposing model files or contract contents:

```json
{
  "status": "UP",
  "service": "dcgaimodel-shadow-inference",
  "feature_version": "dcg-features-v6",
  "policy_sha256": "8f82b058f81ace43c89180803c7ec26ac734b84d0092036a77115688337e1bb6",
  "model_seeds": ["20260826", "20260827", "20260828"]
}
```

The Compose healthcheck uses this endpoint. A readiness failure prevents the
Compose Java container from starting, while failures after readiness remain
handled by Java's fail-open, logging-only shadow observer.

## Frozen inputs and execution path

At startup the service reads, hashes, and loads these tracked files into
immutable in-memory state. The request path performs no file writes, model
training, scaler fitting, or artifact saving.

| Input | SHA-256 |
|---|---|
| `data/inference/frozen-v9/policy-packs-v5-compositional.json` | `8f82b058f81ace43c89180803c7ec26ac734b84d0092036a77115688337e1bb6` |
| seed 20260826 normal-family-split model | `5da2fedbee5d1b3c84c79cb75e2cd10c0b3462066b66571570e93fb7ccd84988` |
| seed 20260827 normal-family-split model | `bd464322c272b8ec1d5ab88605022d4a48e3738b63ab1791a4c7e56f37222ac8` |
| seed 20260828 normal-family-split model | `24ec9e4e758370093c228af34e3b05ac65820b3fd413ca80a269206ce1edd2c0` |

The policy snapshot is byte-identical to the pinned policy file used by the V9
artifacts. Startup also checks every model's recorded V9 training-dataset,
training-oracle, and policy identities.

The request path composes only the established components:

1. `ApprovedPolicyContexts::from_json` resolves the requested declared pack;
2. `SchemaChangeFeatureV6Extractor::extract` produces and validates exactly 76
   features;
3. `ThreeWayModelArtifact::from_slice` restores each model and its 76 persisted
   means and 76 persisted standard deviations from authenticated artifact
   bytes;
4. `StandardScaler::transform_vector` applies the frozen statistics;
5. `ThreeWayCompatibilityModel::predict` returns the three probabilities and
   label.

The legacy `InferenceRuntime` is not referenced by the service.

## Run locally

The artifact root is explicit so the binary cannot silently search for or
select a different model or policy file:

```bash
cargo run -- serve-shadow-inference \
  --artifact-root . \
  --bind 127.0.0.1:8080
```

Artifact and policy paths below that root are fixed in code, and every file is
rejected unless its SHA-256 matches the frozen identity above.
