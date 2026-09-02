# Frozen external V9 evaluation

This protocol tests whether a frozen V9 three-way model transfers to independently sourced contract transitions. It is intentionally separate from V5–V9 corpus generation: it never retrains the model, refits its scaler, or adds records to the V9 dataset.

The Stripe/Kubernetes/Vega track is closed. Its final claims, sourcing yield,
non-deployment purpose, backlog boundary, and complete evidence ledger are in
[`external-v9-v10-evaluation-closeout.md`](external-v9-v10-evaluation-closeout.md).

## Closed sourcing finding: simple root-property removal

Further public sourcing is closed for the specific `FIELD_REMOVED` structural
pattern represented by a single simple or optional property removed from the
root object. Two unrelated public histories independently map to V9's
`remove-first-root-property` generator variant:

- Stripe `card.iin` (`v2348` to `v2349`) has policy-free V6 distance 2 from
  V9. Its 1,485 nearest records comprise 99 V9 families and 15 structural
  shapes across 15 policy packs; every nearest record uses
  `remove-first-root-property`.
- Kubernetes `ObjectMeta.clusterName` (`v1.24.0` to `v1.25.0`) has
  policy-free V6 distance 0, 12 exact full-V6 matches, and 60 nearest V9
  records. Its nearest V9 records also use `remove-first-root-property`.

The machine-readable evidence is
`data/external/field-removed-root-pattern-saturation-v1.json`, backed by the
source-specific feature diagnostics named there. Do not repeat Stripe,
Kubernetes, or other public-source searches merely to acquire more examples
of this root-level pattern: its V9 coverage ceiling is established.

This is **not** a finding that all `FIELD_REMOVED` transitions are saturated.
Nested-property removal, required-property removal, and removal combined with
another structural change remain untested external variants. Those variants
may justify a separately scoped source search, with their own provenance and
overlap audit.

## FORWARD/FULL optional-field benchmark setup (not executed)

The next optional-field benchmark is defined as a separate direction-specific
artifact; it must not be appended to or interpreted as part of the `BACKWARD`
V9 corpus. Its planned scope is:

- valid single optional-root-field additions under `FORWARD` and `FULL`;
- explicit open-object and closed-object structural provenance;
- the unchanged approved policy-pack file, with each result stratified by
  compatibility mode, policy pack, structural profile, and mutation variant;
- family-isolated train/validation/test and challenge reservations only after
  a fresh direction-specific conformance audit demonstrates useful class
  variation;
- separate dataset, conformance, readiness, model, and evaluation artifacts,
  with no scaler reuse or record mixing with V9.

Setup status: **defined, not run**. No JAR call, label, corpus generation,
training, inference, or calibration change is authorized by this setup entry.

## FORWARD/FULL optional-field conformance decision

The separately scoped conformance run is complete under the unchanged pinned
JAR and V5 compositional policy-pack file. Its persisted decision is
`data/experiments/forward-full-optional-field-conformance/label-variation-decision-v1.json`.
The apparent addition/removal wording discrepancy is resolved by
`data/experiments/forward-full-optional-field-conformance/mutation-direction-audit-v1.json`:
all 6,000 staged invocations are genuine base-to-candidate optional additions.
The JAR reports them as `[FORWARD] Field removed` because FORWARD evaluation
deliberately diffs candidate against base.
Across both open and closed root-object profiles, every accepted `FORWARD` and
`FULL` optional-field addition was `BREAKING`; JAR runtime rejections were
preserved but never treated as labels. Therefore the label-variation gate is
**NO-GO**: no family-isolated corpus, `benchmark_ready=true` corpus status,
or three-seed evaluation was produced. The decision is intentionally separate
from the BACKWARD V9 corpus and its artifacts.

That NO-GO remains the historical result for the former reversal-only engine
and its old pinned executable. It is not the status of the later V4-P0
direction-aware track. After the engine gained an explicit profile-aware
FORWARD optional-addition rule, the separately frozen `fee3759a...` JAR showed
SAFE open-consumer and BREAKING closed-consumer outcomes. The resulting
family-isolated corpus, runner-enforced readiness gate, and binary three-seed
evaluation are documented in `docs/forward-full-optional-field-v4-p0.md` and
must not be mixed with BACKWARD V9.

## TYPE_CHANGED external pre-screen: Vega (no accepted rows)

The Stripe and Kubernetes external candidate pools each contain zero
`TYPE_CHANGED` transitions. A scoped third-source search therefore examined
the public [`vega/schema`](https://github.com/vega/schema) version history,
whose versioned Vega and Vega-Lite JSON schemas include documented major-format
migrations. The source is BSD-3-Clause licensed and is preserved from repository
snapshot `8feea15c3a84d4261887f5c7e0cbb74909f1c8a2` under
`data/external/vega-schema-type-change-v1/source-repository`.

`v2.7.0` and `v3.0.0` are versioned schema filenames, not Git refs. Corrected
provenance records the latest commits affecting those files:
`e5be017ddb189b5409cf08907ca7ddc231279cfe` for
`vega-lite/v2.7.0.json` and `48e92bf221eaa44571d00485f1af8d910071ae1f`
for `vega-lite/v3.0.0.json`. Their blobs are resolved at those commits and
cross-checked against the pinned snapshot. Full Git history is now mandatory;
the pre-screen rejects shallow clones.

Nine major-version boundaries were pre-screened conservatively. Whole-component
adds/removes, metadata-only changes, compound changes, and unmapped single
changes were excluded. Two single-operation `TYPE_CHANGED` candidates survived,
both from Vega-Lite `v2.7.0` to `v3.0.0`: `ConditionalPredicate<ValueDef>.value`
and `ConditionalSelection<ValueDef>.value` widened from
`number|string|boolean` to also permit `null`.

Both candidates have policy-free V6 structural distance 1 from V9, no exact
full-V6 match, and 75 nearest V9 records. Their nearest V9 variant is
`change-first-supported-root-type`; the sole differing structural coordinate is
`type_widening`. Under the established distance-at-most-one exclusion rule,
both candidates are rejected. Thus this search produces zero non-overlapping
`TYPE_CHANGED` rows and external clean coverage remains zero.

The persisted audit trail is:

- `data/external/vega-schema-type-change-v1/vega-type-change-prescreen-v2.json`;
- `data/external/vega-schema-type-change-v1/vega-type-change-candidates-v4.json`;
- `data/external/vega-schema-type-change-v1/diagnostics-v2/`;
- `data/external/vega-schema-type-change-v1/vega-type-change-overlap-projection-v2.json`;
- `data/external/vega-schema-type-change-v1/traces-v2/conditional-predicate-raw-trace-v2.json`;
- `data/external/vega-schema-type-change-v1/traces-v2/conditional-selection-raw-trace-v2.json`; and
- `data/external/vega-schema-type-change-v1/provenance-integrity-correction-v2.json`.

`canonical-provenance-freeze-v1.json` freezes the v2 pre-screen, v4 candidate
manifest, and v2 overlap projection by SHA-256. The former v1 records remain
under `provenance-bug-archive-v1/` for auditability only and are not an input
to any current sourcing, projection, trace, or evaluation path.

This was a pre-screen only: no JAR was invoked, no compatibility label was
assigned, no model inference was run, and calibration was unchanged. No full
external extraction is approved from these two rejected candidates.

The two candidates are a sparse sample. Under the existing sparse-stratum
policy they provide traceability only: they cannot support a model-accuracy or
`TYPE_CHANGED` saturation conclusion. Read together with the independently
sourced `FIELD_REMOVED` traces, they are only a directional signal about V9's
feature-space coverage—not proof of broader external generalization.

### TYPE_CHANGED sourcing: closed for now

Further sourcing for additional examples of the same root-level
type-widening shape is closed. The Vega candidates are already rejected at
policy-free structural distance one, so collecting more widening examples at
that distance would not improve clean coverage. A future, separately approved
task may target genuinely different sub-patterns only: nested type changes,
array-to-scalar or scalar-to-array changes, and numeric-to-string or
string-to-numeric changes. It must begin with fresh provenance and the
existing identity/exact-feature/near-structural overlap gates; it must not
reuse the archived v1 Vega artifacts.

## Required manifest

Provide a JSON manifest with this shape. Every transition carries its actual provenance: source name, source kind, URL, base and candidate revisions/releases, license, source path, and stable contract family. All are mandatory. Internal DCG seed or conformance fixtures do **not** qualify as an external source.

```json
{
  "format_version": "dcg-external-transition-manifest-v3",
  "summary": {
    "total_transitions": 1,
    "mutation_id_counts": { "FIELD_REMOVED": 1, "TYPE_CHANGED": 0 },
    "limitations": ["TYPE_CHANGED has zero examples in this source selection."],
    "preserved_source_files": [
      {
        "revision": "release-2026-07",
        "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "blob_sha1": "cccccccccccccccccccccccccccccccccccccccc",
        "source_path": "orders/v1-to-v2.json",
        "audit_copy_path": "sources/release-2026-07.json"
      }
    ]
  },
  "transitions": [
    {
      "record_id": "external-001",
      "source": "example-public-contract-history",
      "source_kind": "public-version-history",
      "source_url": "https://example.org/contracts/releases",
      "base_revision": "release-2026-07",
      "candidate_revision": "release-2026-08",
      "base_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
      "candidate_commit": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "base_blob_sha1": "cccccccccccccccccccccccccccccccccccccccc",
      "candidate_blob_sha1": "dddddddddddddddddddddddddddddddddddddddd",
      "source_license": "Apache-2.0",
      "source_path": "orders/v1-to-v2.json",
      "family_id": "independent-family-001",
      "contract_id": "orders-api",
      "old_version": "1.0.0",
      "new_version": "1.1.0",
      "policy_pack": "baseline",
      "mutation_id": "FIELD_REMOVED",
      "base_schema": { "type": "object", "properties": { "id": { "type": "string" } } },
      "candidate_schema": { "type": "object", "properties": {} }
    }
  ]
}
```

Use `public-version-history` or `public-schema-repository` only with an `http://` or `https://` URL. `manual-unrelated` is accepted for independently authored unrelated-domain transitions and must use a stable `manual://` provenance URI. The manifest is local input and may contain raw schemas. Preserve the two original source files separately, addressed by their recorded revisions. The generated report records provenance, schema fingerprints, and oracle evidence, not the raw schemas.

The V3 manifest also requires a `summary` object with the total count, per-mutation counts, explicit limitations, and preserved source-file records. Its positive counts must match the transition rows; zero-count categories may be kept to document missing coverage.

## No-oracle preflight

Before the JAR is allowed to label any external record, validate the manifest and apply the V9 source/family/pair/exact-feature/near-structural-feature contamination gates. This command only reads the JAR to hash it; it does not launch Java or run a model.

```bash
target/release/dcgaimodel preflight-external \
  --manifest data/external/stripe-openapi-v3/stripe-openapi-external-manifest-v3.json \
  --v9-dataset data/generated/dcg-oracle-production-v9-structural-balanced.json \
  --jar data/oracle-binaries/backward-v10/contract-cli-c00da951bac88d2b.jar \
  --policy-packs data-contract-governance/contracts/policy-packs-v5-compositional.json \
  --output data/external/stripe-openapi-v3/v9-preflight-report.json
```

## Run

Build the release binary first, then select one persisted V9 `normal-family-split` model. The example below uses the first V9 seed. Repeating it once per frozen seed gives a seed-aware transfer report without changing any model.

The original V9 executable (`809b25e6...677dc`) is unavailable. Model-scored
external work therefore uses the separately identified BACKWARD V10 binary
only through the runner-enforced dual-provenance gate. The gate requires this
exact registered tuple before any oracle invocation or model inference:

- V9 training dataset:
  `fa09e645480737ba856940778264284d8e832f4532e989c92a259320cfe05ad3`;
- V9 training oracle:
  `809b25e627e43f847f00a0ce87dcde33ad359006bf22be403e26d662274677dc`;
- BACKWARD V10 execution oracle:
  `c00da951bac88d2be245e08917e6fc62f55abf3a50aad2dea67178a82285f6b6`;
- pinned policy pack:
  `8f82b058f81ace43c89180803c7ec26ac734b84d0092036a77115688337e1bb6`;
- equivalence audit:
  `c090f38c178d5652b788501ec7a26919ec24bd215a32d344dbeb2d9392634793`.

The audit file is not trusted merely because its JSON claims success: its
bytes must match the registered audit hash, and its internal identities,
68,820-record coverage, zero mismatch/rejection counts, negative-control
result, and non-interchangeable identity declaration are all checked.

```bash
cargo build --release

target/release/dcgaimodel evaluate-external \
  --manifest data/external/independent-transitions.json \
  --v9-dataset data/generated/dcg-oracle-production-v9-structural-balanced.json \
  --model data/experiments/v9-multiclass-cpu-100e/models/seed-20260826-normal-family-split.json \
  --jar data/oracle-binaries/backward-v10/contract-cli-c00da951bac88d2b.jar \
  --policy-packs data-contract-governance/contracts/policy-packs-v5-compositional.json \
  --oracle-equivalence-audit data/oracle-binaries/backward-v10/behavioral-equivalence-audit-v1.json \
  --workspace data/oracle-staging/external-backward-v10 \
  --output data/experiments/external-backward-v10/seed-20260826-report.json
```

The command rejects overwriting an existing report. Use a separate empty workspace per invocation when running multiple seeds.

## Non-negotiable gates

Before scoring, the evaluator:

- confirms the selected model is a V6 three-way `normal-family-split` artifact;
- verifies the model's V9 dataset/training-oracle identities, the V10 execution
  identity, the shared policy identity, and the registered equivalence-audit
  bytes and contents before reaching any scored transition;
- labels every submitted transition with the pinned JAR in `BACKWARD` mode;
- rejects invalid-oracle records (exit code `2`), V9 source/family overlap, duplicate external pairs, exact V9 pair overlap, exact model-input overlap, and near V9 structural-feature overlap (Hamming distance at most one by default);
- limits policy packs to the audited V9 pack set. A new policy pack needs a separately conformed benchmark rather than being silently mixed into this evaluation.

Every V4 report persists a `dual_provenance` object containing the training
dataset, training oracle, execution oracle, policy pack, and audit hashes. The
runner calls `require_scored_dual_provenance` before returning the report; the
same public guard is the required entry point for any future promotion path.
There is currently no separate external-results promotion command.

This unblocks model-scored Stripe, Kubernetes, and Vega follow-ups under the
single registered V9-training/BACKWARD-V10-execution tuple, subject to their
ordinary manifest, contamination, schema-validity, and class-support gates.
Every other training/execution oracle pairing remains blocked until it has its
own exhaustive audit and explicit approved-pairing registry entry.

Records rejected by an overlap gate retain their JAR evidence in the report but are excluded from metrics. Oracle-rejected records are also excluded; they are not feature-extracted or inferred. A malformed manifest or an unextractable V6 feature for an oracle-accepted transition fails the run rather than producing a partially trusted report.

## Report interpretation

The JSON report includes JAR/policy/dataset hashes, each record's provenance and gate result, and `SAFE`/`WARNING`/`BREAKING` confusion matrices for the scored records:

- overall;
- each mutation;
- each policy pack; and
- each mutation × policy stratum.

This is a transfer evaluation, not an operational approval. A meaningful report needs independently sourced families with sufficient class support in the individual strata. The deterministic JAR remains authoritative regardless of the model result.
