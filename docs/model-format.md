# Model artifact format

Artifacts are portable JSON files with format identifier `dcgaimodel-artifact-v1`. They store:

- a distinct model version and `dcg-features-v1` feature version;
- model kind and classification threshold;
- every layer's weights, biases, and activation;
- fitted scaler means and standard deviations;
- optimizer/training/split metadata and dataset identifier.

Loading validates format and feature versions, model kind, activation names, layer dimensions, finite weights/biases, threshold range, scaler dimensions and positive finite scales, and required metadata. Malformed or incompatible data returns a typed error; it is never trusted merely because it is JSON.

`InferenceRuntime` only loads, extracts/validates features, applies the persisted scaler, and performs a forward pass. It does not train, backpropagate, update weights, or fit preprocessing.
