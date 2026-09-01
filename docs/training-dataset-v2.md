# Training dataset V2

`dcg-features-v2` is a 28-value, label-free schema-pair feature contract for oracle-backed DCG training data. It does not change the meaning of legacy `dcg-features-v1`; prepared datasets and model artifacts persist their exact version and reject incompatible dimensions.

V2’s exact ordered fields are: `field_count`, `fields_added`, `fields_removed`, `fields_changed`, `schema_depth`, `required_field_count`, `required_fields_added`, `required_fields_removed`, `type_changes`, `type_widening`, `type_narrowing`, `enum_values_added`, `enum_values_removed`, `enum_definitions_added`, `enum_definitions_removed`, `constraint_tightened`, `constraint_relaxed`, `conditional_changed`, `one_of_changed`, `any_of_changed`, `all_of_changed`, `not_changed`, `additional_properties_tightened`, `additional_properties_relaxed`, `policy_baseline`, `policy_strict`, `policy_relaxed`, and `policy_other`. They describe only the schema pair and resolved policy; none is an oracle decision or derived target.

The canonical oracle label is `safe`, `warning`, or `breaking`. Binary training derives `breaking = 1` only for BREAKING; SAFE and WARNING are non-breaking. Three-way compatibility training uses a one-hot target in that exact order, a three-output Softmax head, and fused Cross-Entropy gradients. The binary and three-way pipelines both fit their scaler only on training families.

Generated records retain source/family/split-group IDs, policy pack, mutation identifier, oracle JAR hash, captured result, direction, generator version/seed, label authority, and a SHA-256 canonical pair fingerprint. JSON-object key order does not affect this fingerprint; duplicate pair/policy/oracle combinations are rejected. Grouped splitting keeps all variants of one base schema in exactly one partition; the scaler is fit only on training data.

`PreparedDcgDataset::statistics()` exposes class, family, policy, mutation, and feature-version distribution. `training_readiness()` reports structural issues such as missing classes, inadequate families, or split failure. This is training-ready infrastructure, not a production-model claim.

For a bounded three-way experiment, run `cargo run -- train-three-way --input <prepared-dataset.json> --epochs <n> --batch-size <n> --learning-rate <rate> --seed <seed>`. It refuses datasets that fail the structural three-way readiness gate and reports only grouped train/validation/test results; it does not turn a small smoke dataset into a quality claim.
