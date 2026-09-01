# AI Engine Roadmap

## Vision

Build a production-quality Deep Learning Framework in Rust from scratch for the Data Contract Governance platform.

The goal is not only to build an AI model, but also to understand every mathematical and engineering component behind modern machine learning frameworks.

---

# Phase 1 — Mathematical Foundations

## Vector

- [x] Create Vector
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

---

## Matrix

- [x] Matrix Structure
- [x] Identity Matrix
- [x] Zero Matrix
- [x] Matrix Addition
- [x] Matrix Subtraction
- [x] Scalar Multiplication
- [x] Matrix Multiplication
- [x] Matrix × Vector
- [x] Transpose

---

# Phase 2 — Neural Network Core

## Neuron

- [x] Neuron Structure
- [x] Weights
- [x] Bias
- [x] Forward Pass

---

## Layer

- [x] Dense Layer
- [x] Batch Forward Pass

---

## Activations

- [x] ReLU
- [x] Sigmoid
- [x] Tanh
- [x] Softmax

---

## Loss Functions

- [x] Mean Squared Error
- [x] Binary Cross Entropy
- [x] Cross Entropy

---

## Network

- [x] Sequential Network
- [x] Layer Activation Wiring
- [x] Forward Propagation
- [x] Batch Forward Pass

---

# Phase 3 — Backpropagation

- [x] Chain Rule
- [x] Gradient Computation
- [x] Weight Updates

---

# Phase 4 — Optimizers

- [x] SGD
- [x] Momentum
- [x] Adam

---

# Phase 5 — Dataset

- [x] CSV Loader
- [x] Batch Loader
- [x] Train/Test Split
- [x] Normalization

---

# Phase 6 — Training Engine

- [x] Training Loop
- [x] Evaluation and Metrics
- [x] Versioned Model Saving
- [x] Validated Model Loading
- [x] Inference-only Runtime

---

# Phase 7 — AI Engine Integration

- [x] Advisory Data Contract Classifier
- [ ] Policy Prediction
- [ ] Drift Detection
- [ ] Future platform integration (explicitly deferred)

---

# Milestone 4 — Reliability

- [x] Versioned eight-feature DCG contract
- [x] Self-contained realistic DCG fixtures
- [x] Seeded train/validation/test splitting
- [x] Training-only standardization
- [x] Family-aware train/validation/test pipeline with validation-loss history
- [x] Portable prepared-dataset persistence
- [x] Baselines and validation-threshold evaluation reports
- [x] Versioned JSON model artifacts and scaler persistence
- [x] Independent loaded-model inference

# Training Readiness — Feature V2

- [x] Versioned V2 schema-pair feature contract
- [x] Canonical SAFE/WARNING/BREAKING dataset labels and binary breaking view
- [x] Policy-aware oracle generation and family-safe balance controls
- [x] Dataset statistics and structural readiness reporting
- [x] V2 binary smoke-training path
- [x] Correct multiclass Softmax + Cross-Entropy training with a three-output compatibility head
- [x] Portable three-way Softmax artifacts with fixed class ordering and input SHA-256 provenance
- [x] Reproducible CPU multi-seed three-way experiment reports with persisted per-protocol models and aggregate challenge metrics
- [x] Policy-semantic V4 feature contract and held-out-policy identifiability gate
- [x] Approved peer policy profiles, six-pack conformance, benchmark-ready V4 corpus, and three-seed policy-identity evaluation
- [x] V5 59-feature full policy-rule semantic contract, resolved from the pinned JAR's baseline-plus-overrides policy behavior
- [x] Rich independent `CONSTRAINT_TIGHTENED` mutation families, exercised against the pinned JAR
- [x] Approved V5 compositional policy profiles and action-component identifiability gate
- [x] Deterministic eight-worker Rust oracle orchestration, certified byte-for-byte against serial generation
- [x] Fresh 15-pack V5 full conformance and hash-pinned qualified promotions
- [x] Benchmark-ready V9 corpus generation, audit, and three-seed V6 Softmax evaluation under the pinned V5 policy packs
- [x] Family/pair leakage and feature-space near-duplicate review of representative 100% challenge rows
- [x] V5→V9 closeout: optional-field `BACKWARD` behaviour recorded as a qualified SAFE-only traceability case, not a fabricated accuracy score

## Post-closeout research (separate benchmark scope)

- [x] Frozen V9 external-transition evaluation path: pinned-JAR labelling, source/family provenance, V9 overlap exclusion, and inference-only three-way reporting.
- [ ] Execute the separately scoped `FORWARD`/`FULL` optional-field benchmark only after fresh conformance proves label variation; its scope/setup is documented, but no run has started and it must not be mixed with the `BACKWARD` V9 corpus.
- [x] Close further public sourcing for the saturated single simple/optional root-property `FIELD_REMOVED` pattern; nested, required, and compound removal variants remain explicitly untested.
- [x] Pre-screen a third public history for `TYPE_CHANGED`: Vega yielded two conservative type-widening candidates, but both were rejected at V9 structural distance 1, leaving clean external `TYPE_CHANGED` coverage at zero.
- [ ] Evaluate on independently sourced real contract transitions before any operational model-use claim.
