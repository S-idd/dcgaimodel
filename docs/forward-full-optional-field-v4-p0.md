# FORWARD/FULL optional-field V4-P0 evidence

This track is intentionally separate from BACKWARD V9. Its frozen executable
is `fee3759a1f09bad477d2624a5c5342dd4f7c35d3d9089f7b9cbba5333949d923`;
its real pinned policy file is
`8f82b058f81ace43c89180803c7ec26ac734b84d0092036a77115688337e1bb6`.
The FIELD_REMOVED audit fixture was not used for corpus generation, readiness,
training, or evaluation.

## Required-addition gap origin

The gap predates the removal fix. Commit
`e19aa13d98fe128166c309d9ebfcb06e61afdd4f` introduced the original
direction-aware `evaluateForward`. It passed `reversedDiff.requiredAdded()`
into the backward evaluator but never evaluated `diff.requiredAdded()`.
Its own test explicitly asserted `PASS` for an open-consumer genuine required
addition under FORWARD.

An isolated reconstruction at exactly `e19aa13d` changed only that test
expectation to `[FORWARD] Required field added: added`. The test failed with
`actual breakingChanges() == []`. The later removal fix is commit
`18f0f048960536198f5e88574ce227ecd47a0593`; the required-addition fix is
`da93082e1ac6617ddf8355704e78b34c426b8523`. Therefore the gap was present in
the original direction-aware implementation and was not introduced by the
removal fix. Machine-readable evidence is
`data/experiments/forward-removal-mechanism-audit/required-addition-gap-origin-audit-v1.json`.

## Family-isolated corpus

The corpus command directly calls the existing
`src/models/external_evaluation.rs::preflight_external_manifest` before it
constructs `PinnedOracle`. No replacement isolation check was added. The gate
checks V9 source IDs, family IDs, canonical pairs, exact V6 model inputs, and
policy-free structural distance with the established rejection threshold of
one coordinate.

The source prescreen considered 40 Stripe and 238 Vega-Lite families. Four
families retained matched open and closed profiles beyond the V9 threshold:

- Stripe `card`;
- Stripe `payment_intent_next_action`;
- Vega-Lite `Scale`; and
- Vega-Lite `StyleConfigIndex`.

Extracted source fragments used a documented, label-free standalone
projection: document-local `$ref` nodes were replaced on both sides by the
same lint-valid open-object placeholder. This does not create a pairwise
change. Each resulting pair still contains exactly one optional-field
addition. Array variants at distance one were excluded before the JAR; the
final manifest contains 420 accepted family/profile/variant/policy pairs.

The canonical v4 preflight accepted all 420 records, with every nearest V9
policy-free structural distance equal to two. All 28 unique schema/profile/
variant lint preflights passed. FORWARD and FULL then each retained 420 rows:
210 SAFE open-consumer rows and 210 BREAKING closed-consumer rows, with no
runtime rejection. The combined corpus contains 840 rows, four split-isolated
families, 15 policies, four mutation variants overall, and no V9 family or
pair leakage.

## `benchmark_ready` enforcement

`PreparedDcgDataset::require_benchmark_ready` accepts only `Some(true)` and
returns explicit errors for `Some(false)` and `None`. The active binary
three-seed runner calls it before standard-record selection, output-directory
creation, training, or artifact persistence.

The test
`models::experiment::tests::runner_rejects_false_or_missing_benchmark_ready_before_training`
asserts both refusal messages and asserts that the output path remains absent.
It passed independently, and the full Rust suite passed with 298 library tests
and three integration tests (two explicit JAR integration tests remain
ignored by design).

Only after that test passed was the clean isolation report revalidated against
the frozen JAR and pinned-policy hashes. The readiness audit found three
standard families, one challenge family, no family or pair leakage, four
complete policy-counterfactual family/mutation pairs, and no shortcut risk.
The promoted artifact therefore persists `benchmark_ready: true`.

## Three-seed result

The pinned policy file produces no WARNING label for this optional-addition
track. Three-class readiness correctly refuses the corpus because WARNING is
absent; no WARNING row was manufactured. The completed three-seed evaluation
therefore uses the explicit binary BREAKING/non-breaking target.

Seeds `20260826`, `20260827`, and `20260828` each trained on one standard
family (240 rows), validated on one standard family (180 rows), tested on the
third standard family (180 rows), and evaluated the reserved challenge family
(240 rows). Every seed produced:

- test accuracy/F1 `1.0`, confusion `[[90, 0], [0, 90]]`; and
- challenge accuracy/F1 `1.0`, confusion `[[120, 0], [0, 120]]`.

The feature-65 ablation fits only the two label buckets of
`root_optional_fields_added_to_closed_object` from each seed's training
partition. For every seed it learned `0 → non-breaking` and `1 → breaking`,
then reproduced the full model's test confusion `[[90, 0], [0, 90]]` and
challenge confusion `[[120, 0], [0, 120]]` exactly. Its persisted report is
`data/experiments/forward-full-optional-field-family-isolated/binary-three-seed-v1/feature-65-ablation-v1.json`
(SHA-256
`c5e683e3d00be0fbb75a3043f6ddfb4ba9bf9863a5ea394b863017a11600a40e`).

Therefore the full model's 100% accuracy on this corpus is fully explained by
a single directly readable input feature. This benchmark confirms correct
mechanical wiring of the engine, feature extractor, splits, and evaluator; it
cannot demonstrate model generalization beyond feature 65 or distinguish a
learned general rule from trivial recovery of that feature.

All canonical paths and SHA-256 values, including the three model artifacts,
are frozen in
`data/experiments/forward-full-optional-field-family-isolated/canonical-artifact-freeze-v1.json`.
