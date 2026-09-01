# Changelog

All notable changes to this project will be documented in this file.

This project follows a milestone-based development process. Every completed sprint records:
- Features implemented
- Files modified
- Tests added
- Git commit summary

---

# Version 0.1.0 (Development)

## Repository and Community Infrastructure

- Added GitHub Actions for Rust formatting, Clippy, tests, and scheduled `cargo audit` checks.
- Added Dependabot configuration for Cargo and GitHub Actions dependencies.
- Added Apache-2.0 licensing, security reporting guidance, a code of conduct, issue forms, a pull-request template, and code ownership.
- Declared Rust 1.88 as the minimum supported version and added shared toolchain, rustfmt, Clippy, EditorConfig, and Git attribute settings.
- Excluded generated corpora, oracle staging trees, external repository checkouts, and the separately maintained governance repository from the Rust repository.

## Milestone 4 — Reliability, Persistence, and Inference Readiness

### Features

- Added the versioned `dcg-features-v1` feature contract and strict raw-feature validation.
- Added self-contained realistic DCG contract-change fixtures and prepared training records.
- Added seeded train/validation/test split support and validated persisted scaler statistics.
- Added prepared-dataset JSON artifacts with source, family/split-group, contract version, policy-pack, label, and feature-version metadata.
- Added family-aware partitioning, per-epoch validation loss tracking, and a unified train/validation/test pipeline.
- Added majority and deterministic breaking-count baselines plus arbitrary threshold evaluation.
- Added pipeline-level validation threshold and isolated-test baseline reports.
- Added `dcgaimodel-artifact-v1` JSON model artifacts with network, scaler, and training metadata validation.
- Added inference-only `InferenceRuntime` and structured `PredictionResult` output.
- Added an end-to-end realistic-fixture train → save → load → infer integration test.

### Safety boundary

`dcgaimodel` predictions are advisory. Deterministic DCG compatibility and policy enforcement remain authoritative. An ML prediction must not override a deterministic breaking-change or policy violation result.

### Dependencies

- Added `serde` and `serde_json` for an explicit, portable, validated JSON artifact format.

### Status

Completed — verified by the Milestone 4 quality gate.

## Training Readiness — Feature V2 and Oracle Dataset Expansion

- Added `dcg-features-v2`, a versioned 28-feature structural and policy-context schema-pair contract without oracle-label leakage.
- Added canonical SAFE/WARNING/BREAKING storage with a centralized derived binary BREAKING target.
- Added V2-aware prepared-data and model-artifact feature-version validation while preserving V1 semantics.
- Added deterministic broad seed sampling, three-policy oracle generation, balanced outcome retention, dataset statistics, duplicate-ID rejection, and readiness reporting.
- Added a grouped three-way compatibility workflow with one-hot SAFE/WARNING/BREAKING targets, a three-output Softmax head, fused Cross-Entropy gradients, and per-epoch validation loss.
- Added a small V2 binary smoke run only; no production-quality training claim is made.

## Three-Way Artifact Persistence and Reproducible CPU Evaluation

- Added a separate `dcgaimodel-three-way-artifact-v1` format for SAFE/WARNING/BREAKING Softmax models, preserving the old binary artifact format unchanged.
- Persisted the fixed SAFE/WARNING/BREAKING output order, training-only scaler, full training metadata, and SHA-256 identities for the dataset, oracle JAR, and policy-pack file.
- Added a CPU multi-seed experiment runner that verifies the supplied JAR and policy pack against generated-record provenance before training.
- Added per-seed normal, held-out-policy, held-out-mutation, `ENUM_VALUE_ADDED` structural-variant, and same-mutation/different-policy reports with full loss histories and aggregate confusion matrices.
- Recorded one-class mutation holdouts as traceability-only and the full policy-sensitive `ENUM_VALUE_ADDED` holdout as structurally unidentifiable, rather than reporting either as a valid generalization score.

## Policy-Holdout Generalization Safeguards

- Added `dcg-features-v4`, which replaces policy-name identity features with the declared `ENUM_VALUE_ADDED` IGNORE/WARNING/BREAKING disposition parsed from the approved policy-pack JSON.
- Added a V4 held-out-policy identifiability gate. A policy profile absent from training is recorded as structurally unidentifiable instead of being presented as a supervised generalization score.
- Confirmed the current three-pack configuration has one unique semantic profile per pack; a measurable policy-identity holdout requires separately approved peer profiles and new pinned-input conformance evidence.

## V4 Policy-Identity Benchmark Completion and V5 Feature Foundation

- Added the approved six-pack peer-policy configuration, preserving the original policy-pack file unchanged.
- Captured 58,086 six-policy pinned-JAR conformance checks and created 120 explicit, hash-pinned qualified promotions. Policy-variable cells remain unpromoted.
- Generated and audited the V4 production corpus: 17,994 JAR-labelled records across 392 independent families, with 300 standard and 92 family-isolated challenge families. It reports `benchmark_ready=true` with no family or oracle-pair leakage.
- Completed the V4 three-seed CPU Softmax experiment, persisting 39 models and aggregate reports. Held-out named-policy accuracy is 99.66%–99.76%; this is policy-name generalization over known peer semantics, not zero-shot policy-behavior generalization.
- Added `dcg-features-v5`, a 59-feature contract that replaces V4's enum-only policy context with 24 resolved IGNORE/WARNING/BREAKING actions spanning all eight JAR rule IDs.
- Matched feature-policy resolution to the executable JAR's baseline-plus-pack-overrides behavior; feature extraction remains independent of oracle exit code and compatibility label.
- Expanded `CONSTRAINT_TIGHTENED` generation into 15 bounded, independently identified variants: numeric minimum/maximum/multiple-of, string minimum/maximum length and pattern, object property bounds, array item/uniqueness constraints, and mixed forms.
- Added an opt-in pinned-JAR test that stages every hard candidate and verifies its actual `BACKWARD`/`baseline` label. The reviewed run used JAR SHA-256 `809b25e627e43f847f00a0ce87dcde33ad359006bf22be403e26d662274677dc` and policy-pack SHA-256 `65a5f29addcf999683b18c888c6d60a082023fa0514c6b1b3881fc7512d5039a`; all 15 were valid and `BREAKING`.
- Added the separately pinned V5 compositional policy file, with the original six profiles plus a complete 3×3 `ENUM_VALUE_ADDED` × `CONSTRAINT_TIGHTENED` IGNORE/WARNING/BREAKING matrix. It has SHA-256 `8f82b058f81ace43c89180803c7ec26ac734b84d0092036a77115688337e1bb6`.
- Updated V5 held-out-policy gating: an unseen full vector is scored as a compositional test only when each rule/action component is represented independently in training; a missing component remains structurally unidentifiable. This does not inspect labels.
- Executed 18 targeted pinned-JAR checks over the nine compositional profiles (enum addition and constraint tightening per profile). Every result matched the declared SAFE/WARNING/BREAKING action; full three-mode conformance remains required before promotions or corpus claims.
- Added bounded `--oracle-workers` parallelism to Rust generation. Oracle calls for one candidate are executed in isolated processes but consumed in the declared policy order, so quotas, family reservation, labels, and record order stay deterministic.
- Certified the eight-worker implementation against serial generation with the V5 JAR and policy file: two quota-stress replays (135 retained records) and a full-label 20-source replay (5,340 retained records) were byte-identical to their serial references.

## Counterfactual Corpus and Generalization Safeguards

- Added complete-family `standard`/`challenge` dataset roles; challenge records are persisted but excluded from train/validation/test splitting.
- Added bounded per-policy/mutation/oracle-outcome retention for counterfactual corpus generation.
- Added reusable policy × label, mutation × label, and policy × mutation × label shortcut audits with observed oracle-invariant mutation exceptions.
- Added corpus gates for challenge leakage, complete three-policy family/mutation coverage, shortcut risks, and multi-class challenge views.
- Added held-out-policy, held-out-mutation, and same-mutation/different-policy challenge subset protocols.
- Added a bounded oracle-backed counterfactual smoke-generation and audit procedure. This remains a corpus-design readiness check, not operational-generalization evidence.

## Sprint 1 — Project Initialization

### Objective
Create the initial Rust project and establish the project architecture.

### Features
- Initialized Cargo project
- Created module hierarchy
- Added project directories
- Created library entry point (`lib.rs`)
- Created placeholder modules for future development

### Files Added

```
Cargo.toml
src/lib.rs

src/linalg/
src/nn/
src/activations/
src/losses/
src/dataset/
src/preprocessing/
src/training/
src/inference/
src/api/
src/utils/

tests/
```

### Tests

No tests.

### Status

Completed

---

## Sprint 2 — Vector Foundation

### Objective

Implement the first mathematical data structure.

### Features

- Implemented `Vector`
- Added constructor
- Added `len()`
- Added `is_empty()`
- Added `get()`
- Added safe vector addition
- Added custom error handling

### Files Modified

```
src/linalg/vector.rs
src/linalg/errors.rs
```

### Tests Added

- Create vector
- Create empty vector
- Get valid element
- Invalid index
- Vector addition
- Dimension mismatch

### Result

6 Unit Tests Passed

### Status

Completed

---

## Sprint 3 — Vector Subtraction

### Objective

Implement vector subtraction.

### Features

- Added `subtract()`

### Files Modified

```
src/linalg/vector.rs
```

### Tests Added

- Vector subtraction
- Dimension mismatch

### Result

8 Unit Tests Passed

### Status

Completed

---

## Sprint 4 — Scalar Multiplication

### Objective

Implement scalar multiplication.

### Features

- Added `scalar_multiply()`

### Files Modified

```
src/linalg/vector.rs
```

### Tests Added

- Positive scalar
- Zero scalar
- Negative scalar

### Result

11 Unit Tests Passed

### Status

Completed

---

## Sprint 5 — Dot Product

### Objective

Implement the mathematical dot product.

### Features

- Added `dot_product()`

### Files Modified

```
src/linalg/vector.rs
```

### Tests Added

- Dot product
- Dimension mismatch
- Zero vector

### Result

14 Unit Tests Passed

### Status

Completed

---

## Sprint 6 — Vector Magnitude and Normalization

### Objective

Complete vector normalization support for mathematical preprocessing and future model training.

### Features

- Added `magnitude()`
- Added `normalize()`
- Added zero-magnitude error handling

### Files Modified

```
src/linalg/vector.rs
src/linalg/errors.rs
```

### Tests Added

- Vector magnitude
- Zero-vector magnitude
- Single-element magnitude
- Vector normalization
- Zero-vector normalization error

### Result

55 Unit Tests Passed

### Status

Completed

---

## Sprint 7 — Matrix Foundation

### Objective

Implement the core matrix operations needed for neural-network layers and batched computation.

### Features

- Implemented `Matrix`
- Added matrix shape helpers
- Added matrix element access
- Added matrix addition
- Added matrix subtraction
- Added scalar multiplication
- Added transpose
- Added identity matrix creation
- Added zero matrix creation
- Added matrix-vector multiplication
- Added matrix-matrix multiplication

### Files Modified

```
src/linalg/matrix.rs
src/linalg/mod.rs
tests/linalg_integration.rs
```

### Tests Added

- Matrix construction
- Empty matrix
- Irregular matrix rejection
- Matrix element access
- Matrix display
- Matrix addition and subtraction
- Scalar multiplication
- Transpose
- Identity matrix
- Zero matrix
- Matrix-vector multiplication
- Matrix-matrix multiplication
- Integration checks for identity and transpose

### Result

57 Tests Passed

### Status

Completed

---

## Sprint 8 — Neuron Foundation

### Objective

Implement the first neural-network building block.

### Features

- Implemented `Neuron`
- Added weights
- Added bias
- Added input-size helper
- Added linear forward pass

### Files Modified

```
src/nn/neuron.rs
src/nn/mod.rs
```

### Tests Added

- Create neuron
- Forward pass
- Forward dimension mismatch

### Status

Completed

---

## Sprint 9 — Dense Layer Foundation

### Objective

Implement dense-layer computation on top of the neuron primitive.

### Features

- Implemented `Layer`
- Added dense layer construction from neurons
- Added dense layer construction from weights and biases
- Added layer shape helpers
- Added single-input forward pass
- Added batch forward pass

### Files Modified

```
src/nn/layer.rs
src/nn/mod.rs
```

### Tests Added

- Create dense layer from neurons
- Create dense layer from weights and biases
- Empty layer behavior
- Reject mixed neuron input sizes
- Reject weight and bias count mismatch
- Single-input forward pass
- Forward dimension mismatch
- Empty-layer forward pass
- Batch forward pass
- Batch dimension mismatch

### Result

70 Tests Passed

### Status

Completed

---

## Sprint 10 — Activation Functions

### Objective

Implement the activation functions needed for neural-network forward propagation.

### Features

- Added ReLU
- Added Sigmoid
- Added Tanh
- Added Softmax
- Added vector helpers for activation functions
- Added numerically stable sigmoid and softmax implementations

### Files Modified

```
src/activations/mod.rs
```

### Tests Added

- ReLU scalar behavior
- ReLU vector behavior
- Sigmoid scalar behavior
- Sigmoid large-value stability
- Sigmoid vector behavior
- Tanh scalar behavior
- Tanh vector behavior
- Softmax empty input
- Softmax probability sum
- Softmax large-logit stability

### Result

82 Tests Passed

### Status

Completed

---

## Sprint 11 — Loss Functions

### Objective

Implement the loss functions needed for model training and evaluation.

### Features

- Added `LossError`
- Added mean squared error
- Added binary cross entropy
- Added multiclass cross entropy
- Added probability and target validation
- Added distribution validation for cross entropy

### Files Modified

```
src/losses/mod.rs
```

### Tests Added

- Mean squared error
- Mean squared error for equal vectors
- Mean squared error dimension mismatch
- Mean squared error empty input
- Binary cross entropy
- Binary cross entropy exact endpoint handling
- Binary cross entropy dimension mismatch
- Binary cross entropy invalid probability
- Binary cross entropy invalid target
- Cross entropy
- Cross entropy with soft targets
- Cross entropy zero-probability handling
- Cross entropy dimension mismatch
- Cross entropy invalid probability
- Cross entropy invalid target
- Cross entropy invalid prediction distribution
- Cross entropy invalid target distribution

### Result

99 Tests Passed

### Status

Completed

---

## Sprint 12 — Network Wiring and Forward Propagation

### Objective

Connect dense layers and activation functions into a sequential feed-forward network.

### Features

- Added `Activation` enum
- Added linear activation for output layers
- Added activation application helper
- Added `NetworkLayer`
- Added `Network`
- Added `NetworkError`
- Added adjacent-layer shape validation
- Added single-input forward propagation
- Added batch forward propagation

### Files Modified

```
src/activations/mod.rs
src/nn/network.rs
src/nn/mod.rs
```

### Tests Added

- Linear activation enum behavior
- Activation enum ReLU behavior
- Activation enum Softmax behavior
- Create network
- Reject empty network
- Reject empty layer
- Reject incompatible adjacent layers
- Network layer forward with activation
- Sequential forward propagation
- Softmax output layer
- Forward input dimension mismatch
- Batch forward propagation
- Batch forward dimension mismatch

### Result

112 Tests Passed

### Status

Completed

---

## Sprint 13 — MSE Backpropagation

### Objective

Implement the first complete backpropagation path for sequential dense networks.

### Features

- Added activation derivatives for trainable activations
- Added neuron-level gradient application
- Added `LayerGradient`
- Added `NetworkGradients`
- Added MSE backpropagation
- Added chain-rule gradient propagation through hidden layers
- Added gradient-descent weight updates
- Added one-step MSE training helper
- Added explicit unsupported activation handling for Softmax backpropagation

### Files Modified

```
src/activations/mod.rs
src/nn/neuron.rs
src/nn/layer.rs
src/nn/network.rs
src/nn/mod.rs
```

### Tests Added

- Activation derivatives
- Neuron gradient application
- Neuron gradient dimension mismatch
- Single-layer MSE gradients
- Hidden-layer chain-rule gradients
- Backprop target dimension mismatch
- Unsupported Softmax backpropagation
- Network gradient application
- Invalid learning rate
- One-step MSE training update

### Result

122 Tests Passed

### Status

Completed

---

## Sprint 14 — SGD Optimizer

### Objective

Introduce the optimizer abstraction and implement stochastic gradient descent.

### Features

- Added `Optimizer` trait
- Added `OptimizerError`
- Added `Sgd`
- Added learning-rate validation
- Added optimizer gradient step
- Added optimizer-backed MSE training step

### Files Modified

```
src/nn/optimizer.rs
src/nn/mod.rs
```

### Tests Added

- Create SGD optimizer
- Reject invalid learning rate
- Update learning rate
- SGD applies gradients to network
- SGD returns network errors
- SGD MSE training step reduces error

### Result

128 Tests Passed

### Status

Completed

---

## Sprint 15 — Momentum and Adam Optimizers

### Objective

Complete Phase 4 optimizer support.

### Features

- Added Momentum optimizer
- Added velocity state tracking
- Added Adam optimizer
- Added first and second moment tracking
- Added Adam bias correction
- Added optimizer hyperparameter validation
- Added optimizer-backed MSE training helpers

### Files Modified

```
src/nn/optimizer.rs
src/nn/mod.rs
```

### Tests Added

- Create Momentum optimizer
- Reject invalid momentum
- Update Momentum hyperparameters
- Momentum velocity accumulation
- Momentum MSE training step
- Create Adam optimizer with defaults
- Create Adam optimizer with explicit hyperparameters
- Reject invalid Adam hyperparameters
- Update Adam learning rate
- Adam bias-corrected update
- Adam MSE training step

### Result

139 Tests Passed

### Status

Completed

---

# Current Progress

## Mathematical Foundation

### Vector

- [x] Constructor
- [x] len()
- [x] is_empty()
- [x] get()
- [x] add()
- [x] subtract()
- [x] scalar_multiply()
- [x] dot_product()
- [x] magnitude()
- [x] normalize()

### Matrix

- [x] Matrix structure
- [x] Identity matrix
- [x] Zero matrix
- [x] Matrix addition
- [x] Matrix subtraction
- [x] Scalar multiplication
- [x] Matrix multiplication
- [x] Matrix-vector multiplication
- [x] Matrix transpose

### Neural Network

- [x] Neuron
- [x] Layer
- [x] Activations
- [x] Loss Functions
- [x] Network
- [x] Forward Propagation
- [x] Backpropagation
- [x] SGD Optimizer
- [x] Momentum Optimizer
- [x] Adam Optimizer
- [ ] Training Loop

---

Last Updated:
Week 1 — Phase 4 Optimizers Complete
