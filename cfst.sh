#!/bin/bash

set -e

echo "========================================="
echo " Creating AI Engine Project Structure..."
echo "========================================="

# Create directories
mkdir -p src/linalg
mkdir -p src/nn
mkdir -p src/activations
mkdir -p src/losses
mkdir -p src/dataset
mkdir -p src/preprocessing
mkdir -p src/training
mkdir -p src/inference
mkdir -p src/api
mkdir -p src/utils
mkdir -p tests

# Create library entry point
touch src/lib.rs

# -----------------------------
# Linear Algebra Module
# -----------------------------
touch src/linalg/mod.rs
touch src/linalg/vector.rs
touch src/linalg/matrix.rs
touch src/linalg/operations.rs
touch src/linalg/errors.rs

# -----------------------------
# Neural Network Module
# -----------------------------
touch src/nn/mod.rs
touch src/nn/neuron.rs
touch src/nn/layer.rs
touch src/nn/network.rs
touch src/nn/optimizer.rs

# -----------------------------
# Other Modules
# -----------------------------
touch src/activations/mod.rs
touch src/losses/mod.rs
touch src/dataset/mod.rs
touch src/preprocessing/mod.rs
touch src/training/mod.rs
touch src/inference/mod.rs
touch src/api/mod.rs
touch src/utils/mod.rs

echo ""
echo "========================================="
echo " Project Structure Created Successfully!"
echo "========================================="
echo ""

tree
