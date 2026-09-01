# dcgaimodel

[![Rust CI](https://github.com/S-idd/dcgaimodel/actions/workflows/ci.yml/badge.svg?branch=WEEK1)](https://github.com/S-idd/dcgaimodel/actions/workflows/ci.yml)
[![Dependency Security Audit](https://github.com/S-idd/dcgaimodel/actions/workflows/security-audit.yml/badge.svg?branch=WEEK1)](https://github.com/S-idd/dcgaimodel/actions/workflows/security-audit.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

`dcgaimodel` is a Rust machine-learning library for advisory Data Contract Governance (DCG) risk analysis.

Deterministic DCG compatibility and policy enforcement remain authoritative. A model prediction must never override a deterministic breaking-change or policy-violation result.

## Milestone 4 workflow

1. Extract `dcg-features-v1` from self-contained DCG contract-change inputs.
2. Persist prepared records with source, family/split-group, contract version, policy pack, labels, and feature-version metadata.
3. Create a seeded family-aware train/validation/test split.
4. Fit `StandardScaler` on the training partition only, then reuse it everywhere.
5. Train with per-epoch validation loss, evaluate transparent baselines and validation thresholds, then test once.
6. Persist the network, feature version, scaler, threshold, and training metadata as a versioned JSON artifact.
7. Load that artifact in `InferenceRuntime` for inference without retraining or refitting.

See [feature specification](docs/feature-specification.md), [training](docs/training.md), [evaluation](docs/evaluation.md), [prepared datasets](docs/prepared-dataset-format.md), [model format](docs/model-format.md), and the [Milestone 4 audit](docs/milestone-4-audit.md).

For oracle-backed training work, see [training dataset V2](docs/training-dataset-v2.md), [oracle-backed generation](docs/oracle-data-generation.md), the [V5→V9 benchmark closeout](docs/v5-v9-benchmark-closeout.md), and [frozen external evaluation](docs/external-evaluation.md). These provide training-ready infrastructure only; deterministic DCG remains authoritative.

## Development

The repository uses the stable Rust toolchain with Rust 1.88 as its declared minimum supported
version.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --all-features --locked
```

The optional oracle-backed workflows require a separately built and hash-pinned checkout of
[`S-idd/data-contract-governance`](https://github.com/S-idd/data-contract-governance). Generated
corpora, oracle staging directories, and external source checkouts are intentionally excluded from
Git; reviewed manifests, reports, and compact evidence remain under `data/`.

## Repository standards

- [Contributing guide](CONTRIBUTING.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security policy](SECURITY.md)
- [Apache-2.0 license](LICENSE)

GitHub Actions enforce formatting, Clippy, tests, and scheduled dependency auditing. Dependabot is
configured for Cargo and GitHub Actions updates.
