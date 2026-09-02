# Oracle-backed data generation

The Rust generator creates labelled before/after schema pairs from
JSONSchemaBench. It invokes the pinned `contract-cli` JAR for every retained
label; no Markdown rule description is used as a label source.

For a balanced, reproducible corpus across the three installed policy packs:

```sh
cargo run -- generate \
  --input /Users/siddarthkanamadi/Desktop/dcg-training-data/raw/jsonschemabench/data/full-00000-of-00001.parquet \
  --output data/generated/oracle-balanced-v2.json \
  --jar data-contract-governance/contract-cli/target/contract-cli-0.1.0-SNAPSHOT-all.jar \
  --policy-packs data-contract-governance/contracts/policy-packs.json \
  --workspace data/oracle-staging/balanced-v2 \
  --max 200 --seed 42 --per-outcome 50 \
  --dataset-version dcg-oracle-balanced-v2
```

Without explicit `--policy` arguments, generation evaluates every candidate
under `baseline`, `strict`, and `relaxed`. Source selection deterministically
prioritizes independent schemas that can propose enum additions, because the
pinned JAR is the only authority that can turn those pairs into WARNING,
BREAKING, or SAFE records under each policy pack.

`--per-outcome N` keeps at most `N` accepted records per JAR outcome. The
generator continues to stage and check candidates while an outcome is scarce;
records dropped after a class reaches its quota are counted in the run report.
All retained policy variants share the source schema's `family_id` and
`split_group_id`, so the prepared-data split keeps related examples together.

## Counterfactual challenge corpus

For generalization work, do not combine `--per-outcome` with the
counterfactual quota: a global class cap can erase the policy variants needed
to test a source pair fairly. Use the per-policy/mutation/outcome cap and
reserve entire source families for challenge evaluation instead:

```sh
cargo run -- generate \
  --input /Users/siddarthkanamadi/Desktop/dcg-training-data/raw/jsonschemabench/data/full-00000-of-00001.parquet \
  --output data/generated/dcg-oracle-counterfactual-smoke-v1.json \
  --jar data-contract-governance/contract-cli/target/contract-cli-0.1.0-SNAPSHOT-all.jar \
  --policy-packs data-contract-governance/contracts/policy-packs.json \
  --workspace data/oracle-staging/counterfactual-smoke-v1 \
  --max 24 --seed 20260824 \
  --per-policy-mutation-outcome 2 --challenge-family-ratio 0.50 \
  --dataset-version dcg-oracle-counterfactual-smoke-v1

cargo run -- audit \
  --input data/generated/dcg-oracle-counterfactual-smoke-v1.json \
  --seed 20260824
```

`audit` emits the required policy × label, mutation × label, and policy ×
mutation × label tables. It also records observed label-invariant mutations as
exceptions, refuses to call a label-pure avoidable stratum a valid benchmark,
checks standard/challenge family and oracle-pair leakage, and describes the
available held-out-policy, held-out-mutation, and same-mutation/different-
policy challenge views. `benchmark-ready=true` is a corpus-design gate only;
it is not a model-quality claim.

The 50% holdout in this tiny smoke command is solely to guarantee that the
protocol paths are exercised with very few source families. Use a lower,
pre-registered holdout fraction (for example 15–20%) for the larger corpus.

## Coverage-driven production batch

`--max` is always a hard source-family budget. Supplying all four target
options makes the generator reassess the saved-artifact readiness conditions
after every completed family and stop early only when they pass. Otherwise it
uses the full budget and prints the precise blockers.

```sh
cargo run -- generate \
  --input /Users/siddarthkanamadi/Desktop/dcg-training-data/raw/jsonschemabench/data/full-00000-of-00001.parquet \
  --output data/generated/dcg-oracle-coverage-v1.json \
  --jar data-contract-governance/contract-cli/target/contract-cli-0.1.0-SNAPSHOT-all.jar \
  --policy-packs data-contract-governance/contracts/policy-packs.json \
  --workspace data/oracle-staging/coverage-v1 \
  --promotions data/generated/dcg-oracle-promotions-v1.json \
  --max 500 --seed 20260825 --challenge-family-ratio 0.18 \
  --per-policy-mutation-outcome 1 \
  --target-records 3000 --target-standard-families 300 \
  --target-challenge-families 50 --target-counterfactual-pairs 300 \
  --dataset-version dcg-oracle-coverage-v1
```

In coverage-driven mode, the optional policy/mutation/outcome quota applies
per independent family, rather than globally, so it prevents repetitive
variants without preventing family diversity. A label-pure mutation can be
exempted only by a reviewed, identity-pinned promotion manifest generated from
`oracle-conformance`; it cannot be declared on the corpus-generation command.

### Optional-field structural balance

For an open-vs-closed optional-field challenge, use `balanced-optional` rather
than broad sampling. It takes equal label-free source budgets from root objects
that permit or forbid additional properties. The two profile minima are hard
coverage gates: generation cannot report success unless both profiles occur in
both standard and whole-family challenge data. Every generated record persists
its source `root_object_profile` as `open` or `closed`; it is provenance, not a
model label.

These are structural-coverage gates, not a promise of label balance. In the
closed V9 `BACKWARD` benchmark both profiles are oracle-labelled SAFE only, so
they are correctly reported as traceability-only rather than assigned a
three-class score. See [the V5→V9 closeout](v5-v9-benchmark-closeout.md) for
the exact coverage and qualification.

```sh
cargo run --release -- generate \
  --mode FORWARD \
  --source-profile balanced-optional \
  --min-optional-profile-standard-families 150 \
  --min-optional-profile-challenge-families 20 \
  # plus the normal pinned-JAR, policy, coverage, and output options
```

`--mode` accepts exactly one of `BACKWARD`, `FORWARD`, or `FULL`; write and
evaluate a separate corpus for each direction. `FORWARD` and `FULL` must not
be mixed with the `BACKWARD` V9 corpus, its scaler, or its model artifacts.

## Generalization evaluation protocols

After a promotion-aware audit reports `benchmark_ready=true`, run the four
binary evaluation categories from the prepared artifact. The command fits each
scaler on that protocol's standard training partition only. For held-out policy
and mutation trials it removes the requested policy or mutation from all
training, validation, and normal-test records before evaluating the isolated
challenge subset.

```sh
cargo run -- evaluate-protocols \
  --input data/generated/dcg-oracle-production-v1.json \
  --epochs 100 --batch-size 16 --learning-rate 0.05 \
  --seeds 20260826,20260827,20260828
```

It reports normal test metrics, one held-out-policy trial per policy, one
held-out-mutation trial per mutation, and the same-mutation/different-policy
challenge for each supplied split seed. `--seed` may be repeated, while
`--seeds` accepts a comma-separated list; duplicate seeds are evaluated once.
One-label held-out-mutation challenges are retained and printed as unscored
traceability evidence; they are not meaningful generalization metrics. During
generation, whole standard families are additionally reserved for challenge
evaluation when needed to retain every oracle-observed label for a
label-variable mutation family.

For a SAFE/WARNING/BREAKING held-out mutation evaluation, use:

```sh
cargo run -- evaluate-three-way-mutation \
  --input data/generated/dcg-oracle-production-v3-challenge-balanced.json \
  --mutation enum_value_added \
  --epochs 100 --batch-size 16 --learning-rate 0.05 --seed 20260826
```

It trains on standard families without the selected mutation, fits scaling on
that training data only, and reports a three-way accuracy and confusion matrix
on the isolated challenge families.

When the approved policy packs alter only one mutation family, a full
held-out-family policy test is not identifiable: training contains no example
of that policy effect. Use `--variant <mutation-variant>` to hold out one
structural form while retaining other forms of the same mutation in training;
this measures generalization within the policy-sensitive rule without
hard-coding an oracle outcome.

## Reproducible multi-seed three-way CPU experiments

Use the experiment runner for a durable SAFE/WARNING/BREAKING run rather than
interpreting one normal split as a model-quality claim. It creates a distinct
Softmax model artifact for every fitted normal or scored challenge protocol,
and a JSON report with every seed, scaler/model artifact path, full training
and validation loss histories, test/challenge confusion matrices, and SHA-256
identities for the prepared dataset, pinned JAR, and policy-pack file.

```sh
cargo run --release -- run-three-way-experiments \
  --input data/generated/dcg-oracle-production-v3-challenge-balanced.json \
  --output-dir data/experiments/v3-three-way-cpu-100e \
  --oracle-jar data-contract-governance/contract-cli/target/contract-cli-0.1.0-SNAPSHOT-all.jar \
  --policy-packs data-contract-governance/contracts/policy-packs.json \
  --seeds 20260826,20260827,20260828 \
  --epochs 100 --batch-size 16 --learning-rate 0.05
```

The output directory must be new or empty, preventing accidental overwrites.
The command refuses to run if either supplied pinned file has a SHA-256 that
differs from the label provenance inside the prepared dataset. The JSON report
aggregates mean/min/max accuracy and summed confusion matrices only for scored
protocols. One-class mutation holdouts remain `traceability_only`; a full
`ENUM_VALUE_ADDED` holdout is explicitly `structurally_unidentifiable`, because
it removes every policy-dependent example from training. Its per-variant
counterfactual holdouts remain scored and preserve all three labels.

### Policy-holdout identifiability

`dcg-features-v4` removes policy-name one-hot features. Instead, it records the
declared `ENUM_VALUE_ADDED` action from the exact approved policy-pack JSON as
an IGNORE/WARNING/BREAKING context vector. It never reads an oracle exit code
or compatibility label while extracting that context.

The original three-pack configuration cannot support a scored all-policy
holdout: each pack has a unique enum-addition action, so withholding `baseline`
also removes every WARNING example from training. The six-pack V4 configuration
adds independently named peers for the WARNING, BREAKING, and IGNORE profiles.
It has a separate pinned-JAR conformance report, promotion manifest,
benchmark-ready corpus, and three-seed experiment report. Those scores measure
generalization to a new policy *name* whose declared semantics are represented
in training; they do not establish zero-shot generalization to an unseen policy
behavior.

### V5 full policy-rule context

`dcg-features-v5` is the successor contract for future corpora. It retains the
35 label-free V3 structural features and replaces all policy-name features with
24 action features: IGNORE/WARNING/BREAKING one-hot values for each of the
eight pinned-JAR rule IDs. Policy resolution mirrors the JAR exactly: every
pack begins with the executable baseline rules and applies only its own explicit
overrides. The JSON `defaultPack` is used only when an unknown pack is
requested; declared packs do not inherit its custom overrides.

V5 feature extraction remains independent of the oracle result. The current
V9 corpus was newly conformed and generated under the approved V5
configuration before model training; its pinned identities and scoped results
are recorded in [the V5→V9 closeout](v5-v9-benchmark-closeout.md). A changed
JAR or policy-pack file still requires fresh evidence and a separate corpus.

### V5 compositional policy profiles

`data-contract-governance/contracts/policy-packs-v5-compositional.json` is a
new approved policy input. It preserves the original six named profiles and
adds all nine combinations of the already supported IGNORE/WARNING/BREAKING
actions for `ENUM_VALUE_ADDED` and `CONSTRAINT_TIGHTENED`. Every other rule
remains at the pinned JAR baseline. It is intentionally a new file: changing
the V4 policy file would invalidate its conformance evidence and corpus.

For V5 only, a held-out policy with a novel full 24-value action vector can be
scored as a *compositional* behavior test when every individual rule/action
component also occurs in the training policies. A policy requiring an action
that training never observed remains `structurally_unidentifiable`. This gate
uses policy features only; it never uses labels or oracle output.

The composition matrix was first checked directly with the pinned JAR for one
enum addition and one constraint-tightening pair per profile (18 `BACKWARD`
checks). It was then superseded by the hash-pinned full V5 conformance,
qualified promotions, V9 corpus, and three-seed evaluation documented in [the
V5→V9 closeout](v5-v9-benchmark-closeout.md). A new JAR or policy-pack file
invalidates that chain and requires a fresh full matrix before a new corpus is
claimed.

### Deterministic parallel oracle calls

`generate --oracle-workers N` can execute up to `N` independent JAR processes
for the policy checks belonging to one schema candidate. Each process receives
its own staged contract directory. Rust waits for the batch and consumes results
in the declared policy order before it applies fingerprints, quotas, features,
or family reservation, so concurrency changes throughput rather than corpus
semantics. The default remains one worker.

The eight-worker implementation was certified with the V5 JAR and policy file:
two capped 20-source replays and a 5,340-record all-accepted replay were
byte-identical to serial references. Do not raise the worker count or use it
with a changed JAR/policy file without repeating that equivalence check.

### Post-experiment near-duplicate audit

Before interpreting an unusually clean held-out score, run
`audit-challenge-near-duplicates` for a small, representative set of scored
protocols. It independently verifies that no family ID or canonical oracle
pair fingerprint appears in both the allowed standard-role training data and
the reserved challenge data. It also reports exact overlap in the full V6 model
input and a nearest-neighbour distance over the 49 label-free structural V6
coordinates; policy-action coordinates are excluded from that second measure.

```sh
target/release/dcgaimodel audit-challenge-near-duplicates \
  --input data/generated/dcg-oracle-production-v9-structural-balanced.json \
  --output data/experiments/v9-multiclass-cpu-100e/near-duplicate-audit.json \
  --protocol held-out-mutation:constraint_tightened \
  --protocol held-out-mutation-variant:enum_value_added:add-type-preserving-enum-value-0 \
  --protocol held-out-policy:relaxed \
  --protocol held-out-policy:composition-enum-ignore-constraint-breaking \
  --max-samples 5
```

An exact model-input match is not family or pair leakage: independently
generated schemas can project to the same finite feature vector, and named peer
policies can intentionally resolve to identical actions. It does mean that the
corresponding score is feature-space interpolation rather than evidence that
the feature schema separates those examples. The prepared dataset intentionally
does not persist raw schema text, so this command does not claim text-level or
semantic schema similarity.

### Hard `CONSTRAINT_TIGHTENED` candidate families

The generator now proposes independent, valid candidates for numeric
minimum/maximum/multiple-of constraints; string minimum/maximum length and
patterns; array item bounds and uniqueness; object property bounds; and mixed
numeric, string, and array forms. Candidate names include a canonical root
property ordinal and identify the exact structural form; they are provenance,
never model features or labels.

All candidates still go through the executable oracle. The opt-in JAR test
below stages the complete 15-variant fixture through `BACKWARD` mode under the
approved `baseline` pack; it is intentionally separate from ordinary Cargo
tests because the JAR and policy-pack JSON are external pinned artifacts.

```sh
DCG_PINNED_ORACLE_JAR=data-contract-governance/contract-cli/target/contract-cli-0.1.0-SNAPSHOT-all.jar \
DCG_POLICY_PACKS=data-contract-governance/contracts/policy-packs-v4-generalization.json \
cargo test generation::oracle::tests::pinned_jar_labels_every_hard_constraint_variant -- --ignored --nocapture
```

This is a qualified `BACKWARD`/`baseline` check, not an oracle-invariant
promotion. The closed V5/V9 evidence has the required hash-pinned conformance
and promotion chain; any changed oracle executable or policy-pack file must
repeat that work before a new readiness claim.

## Qualified oracle-conformance evidence and promotion

Before the production corpus, run the pinned JAR over every applicable
mutation, every approved policy pack, and `BACKWARD`, `FORWARD`, and `FULL`.
The v2 output first records a separate JAR `lint` preflight for each proposed
pair. Only preflight-accepted pairs enter the compatibility matrix. Every raw
matrix run retains its mutation variant, consumer profile, exit code,
stdout/stderr, and (when rejected) its rejection stage and reason, plus the JAR
and policy-pack SHA-256 values.

```sh
cargo run -- oracle-conformance \
  --input /Users/siddarthkanamadi/Desktop/dcg-training-data/raw/jsonschemabench/data/full-00000-of-00001.parquet \
  --output data/generated/dcg-oracle-conformance-v2.json \
  --jar data-contract-governance/contract-cli/target/contract-cli-0.1.0-SNAPSHOT-all.jar \
  --policy-packs data-contract-governance/contracts/policy-packs.json \
  --workspace data/oracle-staging/conformance-v2 \
  --max 100 --seed 20260826 --min-families 50 --min-policies 3 --min-checks 150
```

For the bounded optional-field go/no-go, derive explicit Open and Closed
consumers from the same sampled source families and run only FORWARD/FULL:

```sh
cargo run -- oracle-conformance \
  --input /path/to/full-00000-of-00001.parquet \
  --output data/experiments/forward-full-optional-field-conformance/matched-v2.json \
  --jar data-contract-governance/contract-cli/target/contract-cli-0.1.0-SNAPSHOT-all.jar \
  --policy-packs data-contract-governance/contracts/policy-packs-v5-compositional.json \
  --workspace data/oracle-staging/forward-full-optional-matched-v2 \
  --source-profile matched-optional --mutation optional_field_added \
  --mode FORWARD --mode FULL --max 3 --seed 20260901 \
  --min-families 3 --min-policies 15 --min-checks 360
```

Matched mode preflights a deterministic candidate pool capped at four times
`--max`, excludes an entire source family if any profile/variant fails, and
only then takes the final `--max` families. All pool preflight diagnostics
remain persisted.

### Oracle-version boundary for the direction-aware track

The frozen direction-aware executable with SHA-256
`fee3759a1f09bad477d2624a5c5342dd4f7c35d3d9089f7b9cbba5333949d923`
is scoped only to new FORWARD/FULL optional-field work. It supersedes
`bfc7603ab425609740be3b2207f298ed0d0c0f816a59841d5aa219ccd6478941`
after the removal-focused control exposed a pre-existing FORWARD-only gap for
genuine required-field additions, and it also supersedes
`cce6fc2fb10ebbbf9bbd940f73477ecf3d0540d52087f7480c0d0bd3f3e6ef94`
within that track because the earlier build did not evaluate genuine
base-to-candidate removals in FORWARD-only mode. It does not replace the
frozen BACKWARD V9 and existing external-sourcing baseline, whose executable
identity remains
`809b25e627e43f847f00a0ce87dcde33ad359006bf22be403e26d662274677dc`.

Do not combine records, promotions, models, readiness evidence, or aggregate
metrics across those hashes. The old binary could not be recovered and is now
permanently frozen as a historical identity: no new invocation may be called
BACKWARD V9. Existing persisted artifacts remain qualified by their recorded
old hash.

The reproducible successor candidate is archived separately at
`data/oracle-binaries/backward-v10/contract-cli-c00da951bac88d2b.jar`, with
SHA-256
`c00da951bac88d2be245e08917e6fc62f55abf3a50aad2dea67178a82285f6b6`.
It was built twice from source commit `35dfa8858199da668893ce8fb5dd0e8cf7f6434f`
using separate worktrees and Maven caches; the outputs matched byte for byte.
It is permanently BACKWARD V10, never V9 and never interchangeable with
`809b25e6...677dc`.

The completed behavioral audit reconstructed all 68,820 preserved V9
base/candidate/policy invocations and verified all 68,820 historical pair
fingerprints before execution. BACKWARD V10 then matched every preserved V9
outcome and every complete stdout stream, with zero rejections, zero nonempty
stderr streams, and zero mismatches. Ten root/nested/array BACKWARD controls
passed. A deliberate shared-code sensitivity probe also distinguished the old
partial-policy fallback (`BREAKING`) from the later direction-aware JAR
(`WARNING`), proving that the audit detects the identified resolver change.

The exhaustive evidence is
`data/oracle-binaries/backward-v10/behavioral-equivalence-audit-v1.json`,
SHA-256
`c090f38c178d5652b788501ec7a26919ec24bd215a32d344dbeb2d9392634793`.
The earlier
`data/oracle-binaries/backward-v10/reconstruction-checkpoint-v1.json` remains
an accurate pre-audit checkpoint rather than being rewritten after the fact.

The persisted control inventory contains one resolver sensitivity probe plus
ten synthetic BACKWARD controls: root optional additions for open, closed,
and omitted `additionalProperties`; a nested optional addition to a closed
object; root optional-field removal; nested required-field removal; genuine
required-field addition; nested field-type change; array-item field-type
change; and nested constraint tightening. All passed with their expected
SAFE/BREAKING result and exact rule-message fragment. Thus the evidence has
ten controls in addition to the resolver probe, not nine.

This result qualifies the new binary as a behaviorally matched BACKWARD V10
candidate for new, separately versioned work. It does not restore the missing
V9 bytes, authorize relabeling V9 records, or allow V9 and V10 hashes to be
collapsed in a manifest, model, script, or report. The external evaluator now
has an explicit dual-provenance gate: it accepts the audited V9-training/V10-
execution tuple only when the registered audit bytes and all internal coverage
assertions validate, then records both identities in the V4 report. Stripe,
Kubernetes, and Vega model-scored follow-ups are mechanically unblocked under
that one tuple; unaudited pairings remain rejected before inference.

### Removal-severity mechanism audit boundary

The pinned policy file with SHA-256
`8f82b058f81ace43c89180803c7ec26ac734b84d0092036a77115688337e1bb6`
contains 15 packs. All resolve `FIELD_REMOVED` to `BREAKING`; none supplies a
`WARNING` or `IGNORE` override. Repository documentation does not establish
whether that is deliberate governance or an oversight, so an owner decision
is still required rather than inferred here.

The separate file
`data/experiments/forward-removal-mechanism-audit/field-removed-severity-audit-fixture-v1.json`
has SHA-256
`33122df6e7b584935918567d683e8a314afdd53db4499f29791f94fbe9d6fcf6`.
It is mechanism-only evidence for the `RuleSeverity` resolver and makes no
claim about production policy behavior. Ordinary `PinnedOracle::new`
construction rejects this identity by hash even if renamed; only the dedicated
mechanism-audit constructor accepts its exact name and hash. The three-way
runner independently rejects the same hash. It is forbidden for corpus
generation, readiness promotion, external evaluation, and benchmark runs.

The addition conformance corpus remains `optional_field_added`-only. Genuine
removal coverage is carried separately by the five-family executable audit at
`data/experiments/forward-removal-mechanism-audit/removal-focused-conformance-v1.json`.

Candidate discovery is automatic but promotion is explicit. After reviewing
the candidate report, select each qualified mode/policy scope separately:

```sh
cargo run -- promote-invariants \
  --evidence data/generated/dcg-oracle-conformance-v1.json \
  --output data/generated/dcg-oracle-promotions-v1.json \
  --select 'field_removed:BACKWARD:baseline=Reviewed pinned-JAR evidence'

cargo run -- audit \
  --input data/generated/dcg-oracle-coverage-v1.json \
  --promotions data/generated/dcg-oracle-promotions-v1.json \
  --seed 20260825
```

For a targeted evidence extension, repeat `--mutation` for the portable
mutation identifiers to review. `promote-invariants --existing <manifest>`
merges non-overlapping, explicitly reviewed scopes into a new manifest and
rejects ambiguous duplicate scopes.

Promotion matching includes mutation, mode, policy, observed label, JAR hash,
and policy-pack hash. A new JAR or policy-pack file therefore makes an old
promotion stale and unusable automatically.
