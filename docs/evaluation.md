# Evaluation

Classification evaluation returns accuracy, precision, recall, F1, a confusion matrix, false-positive count, and false-negative count. If a precision or recall denominator is zero, the metric is defined as `0.0` rather than NaN.

False negatives are especially important for DCG: they are actual breaking changes classified as safe. Overall accuracy alone is not sufficient.

`MajorityClassBaseline` and `breaking_change_heuristic` provide transparent comparison points. The heuristic is deterministic (`breaking_change_count > 0`) and is not an AI model. The primary pipeline evaluates arbitrary configured thresholds on validation data only, then emits `BaselineComparisonReport` for the isolated test partition. It deliberately does not select a production threshold from the small fixture set.

Risk score evaluation reports MSE and MAE. Classification outputs use sigmoid and are bounded scores used by the binary classifier, but this project does not claim statistical probability calibration.
