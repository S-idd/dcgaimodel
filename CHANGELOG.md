# Changelog

All notable changes to this project will be documented in this file.

This project follows a milestone-based development process. Every completed sprint records:
- Features implemented
- Files modified
- Tests added
- Git commit summary

---

# Version 0.1.0 (Development)

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
- [ ] Backpropagation
- [ ] Training Loop

---

Last Updated:
Week 1 — Network Wiring and Forward Propagation
