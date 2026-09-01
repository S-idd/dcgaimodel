# Milestone 4 completion audit

This audit covers the Milestone 4 reliability, persistence, and inference scope. It does not include the future DCG data-generator programme described in the research report.

| Release gate | Evidence | Status |
| --- | --- | --- |
| Versioned feature contract | `dcg-features-v1`, fixed eight-feature ordering, strict raw validation | Complete |
| Realistic self-contained DCG fixtures | Required safe, breaking, enum-policy, and multi-change scenarios | Complete |
| Prepared data | `dcg-prepared-dataset-v1` stores source, family/split group, contract and schema versions, policy pack, features, and labels | Complete |
| Leakage prevention | `DcgPipelineConfig::run` fits one scaler on train only and reuses it for validation/test/inference | Complete |
| Group leakage | `PreparedDcgDataset::split_by_group` assigns complete groups to exactly one partition | Complete |
| Reproducible workflow | Seeded deterministic group ordering; deterministic current model initialization and batch order | Complete |
| Evaluation | Classification/regression metrics, FP/FN, transparent baselines, validation threshold reports | Complete |
| Validation tracking | `TrainingHistory` records validation loss after every epoch without weight updates | Complete |
| Persistence | Versioned JSON model and prepared-dataset artifacts; persisted scaler and training metadata | Complete |
| Artifact safety | Explicit validation plus malformed/corrupt/incompatible artifact test coverage | Complete |
| Inference-only execution | `InferenceRuntime` loads, validates, scales, and predicts without retraining/refitting | Complete |
| End-to-end verification | Realistic fixture train → save → load → infer integration test | Complete |
| Spring Boot independence | No Spring Boot, JNI, FFI, HTTP, database, or external-runtime dependency added | Complete |

## Quality gate

The release gate is the repository's `cargo fmt -- --check`, `cargo check`, `cargo test`, and strict Clippy command. The implementation must not be treated as complete unless all four pass.

## Boundaries and next work

The fixtures remain a small, self-contained engineering dataset; they are not production evidence and do not establish a production threshold or calibrated probability. The prepared-dataset JSON format is suitable for this Rust milestone, while the report's future JSONL/Parquet generator, oracle bundle, provenance/licence programme, and 10k/100k corpus are a separate training-data milestone. Deterministic DCG policy remains authoritative.
