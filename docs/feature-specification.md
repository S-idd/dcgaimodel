# DCG feature specification

The canonical schema is `dcg-features-v1` with exactly eight values. Order is a model-compatibility contract and is used unchanged for extraction, splitting, scaling, persistence, loading, and inference.

| Index | Name | Meaning | Valid raw value |
| --- | --- | --- | --- |
| 0 | `field_count` | Fields in the resulting schema | finite, non-negative |
| 1 | `fields_added` | Fields introduced by the change | finite, non-negative |
| 2 | `fields_removed` | Fields removed by the change | finite, non-negative |
| 3 | `type_changes` | Fields whose type changed | finite, non-negative |
| 4 | `compatibility_score` | Compatible / unknown / incompatible | exactly `0.0`, `0.5`, or `1.0` |
| 5 | `semantic_version_ordinal` | `major * 1_000_000 + minor * 1_000 + patch` | finite, non-negative |
| 6 | `breaking_change_count` | Identified deterministic breaking changes | finite, non-negative |
| 7 | `dependent_consumer_count` | Known downstream consumers | finite, non-negative |

Missing values, negative count-like values, NaN, and infinities are invalid. No feature is silently imputed or clamped. `StandardScaler` standardizes all eight raw values after fitting only on training data.
