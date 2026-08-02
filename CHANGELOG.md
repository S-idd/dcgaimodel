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
- [ ] magnitude()
- [ ] normalize()

### Matrix

- [ ] Matrix structure
- [ ] Matrix addition
- [ ] Matrix subtraction
- [ ] Matrix multiplication
- [ ] Matrix transpose

### Neural Network

- [ ] Neuron
- [ ] Layer
- [ ] Network
- [ ] Forward Propagation
- [ ] Backpropagation
- [ ] Training Loop

---

Last Updated:
Week 1 — Mathematical Foundations
