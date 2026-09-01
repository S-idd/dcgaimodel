# V5→V9 Oracle Benchmark Closeout

## Decision

The V5→V9 oracle-backed benchmark-engineering programme is complete for its
declared scope:

- the pinned `contract-cli` JAR in `BACKWARD` mode;
- the approved 15-profile V5 compositional policy-pack file;
- the `dcg-features-v6` label-free feature contract; and
- generated JSONSchemaBench source families with strict family isolation.

This is not a production-deployment approval. The deterministic DCG oracle and
policy engine remain authoritative, and these generated-corpus results do not
establish prospective performance on customer contracts, unseen oracle
behaviour, or a changed JAR/policy-pack file.

## Reproducible evidence

The final artifact is
`data/generated/dcg-oracle-production-v9-structural-balanced.json`:

| Item | Final value |
| --- | --- |
| Records / independent families | 68,820 / 419 |
| Standard / challenge families | 343 / 76 |
| Challenge records | 12,255 |
| Feature contract | `dcg-features-v6` (76 inputs) |
| Corpus-design audit | `benchmark_ready=true` |
| Pinned JAR SHA-256 | `809b25e627e43f847f00a0ce87dcde33ad359006bf22be403e26d662274677dc` |
| Policy-pack SHA-256 | `8f82b058f81ace43c89180803c7ec26ac734b84d0092036a77115688337e1bb6` |

The three-seed CPU experiment used seeds `20260826`, `20260827`, and
`20260828`, a `76 → 12 → 6 → 3` Softmax network, Cross-Entropy, 100 epochs,
batch size 16, and learning rate 0.05. It persisted 66 model/scaler artifacts
with loss histories and input identities.

The authoritative machine-readable outputs are:

- `data/experiments/v9-multiclass-cpu-100e/three-way-experiment-report.json`
- `data/experiments/v9-multiclass-cpu-100e/v6-row-by-row-comparison.json`
- `data/experiments/v9-multiclass-cpu-100e/near-duplicate-audit.json`
- `data/experiments/v9-optional-field-structural-traceability.json`

## Results and bounded claims

All 21 directly comparable challenge rows improved or held steady from V6 to
V9. The normal family split moved from 99.91% to 99.87%, a 0.035 percentage
point difference. Notable V9 aggregates are:

| Protocol | Three-seed mean accuracy | Supported interpretation |
| --- | ---: | --- |
| Normal family split | 99.87% | Generated-corpus family-split performance only. |
| Held-out `CONSTRAINT_TIGHTENED` | 100.00% | Strongest held-out result in this corpus; see overlap audit below. |
| Held-out policy combinations | 99.76%–100.00% | Generalization to policy names and observed action-component combinations, not unseen policy behaviour. |
| Same mutation / different policy | 99.84% | Policy-conditioned interpolation within approved policy actions. |
| Held-out enum structural variants | 100.00% | Structural interpolation only; substantial feature overlap remains. |

The experiment reports multi-class SAFE/WARNING/BREAKING performance only when
the challenge supports it. One-class and zero-support protocols are retained as
traceability evidence and never converted into an accuracy claim.

## Leakage and overlap review

The V9 near-duplicate audit checked four representative clean-score protocols.
All had `family_leakage=false` and `pair_fingerprint_leakage=false`.

| Protocol | Exact full V6 matches | Within one policy-free structural coordinate | Conclusion |
| --- | ---: | ---: | --- |
| Held-out `CONSTRAINT_TIGHTENED` | 0 / 780 | 0 / 780 | The strongest clean-score result; nearest observable structural distances were 4–6 coordinates. |
| Enum variant `add-type-preserving-enum-value-0` | 285 / 420 | 390 / 420 | 100% is largely feature-space interpolation across independent families. |
| Held-out `relaxed` policy | 435 / 817 | 759 / 817 | The policy has equivalent peer behaviour represented in training. |
| Held-out `composition-enum-ignore-constraint-breaking` | 435 / 817 | 759 / 817 | It is behaviour-equivalent to `relaxed` under the approved packs. |

The prepared corpus deliberately stores feature vectors and canonical
pair-fingerprints, not raw schema text. Accordingly, this audit rules out
family and exact-pair leakage and describes model-visible overlap; it does not
claim text-level schema dissimilarity.

## Optional-field addition: resolved scope, not an open failure

The V9 corpus has explicit open/closed root provenance and meets the structural
coverage target:

| Profile | Standard families | Challenge families | Challenge label distribution |
| --- | ---: | ---: | --- |
| `root-open` | 193 | 46 | SAFE: 2,760 |
| `root-closed` | 150 | 30 | SAFE: 1,800 |

Both root-profile challenges are SAFE-only under the pinned JAR, `BACKWARD`
mode, and approved policy packs. They are therefore `traceability_only`:
family-isolated and correctly recorded, but unsuitable for a three-class
accuracy metric. The former V6 optional-field score (33.28%) is not comparable
to V9; V9 correctly treats the complete mutation holdout as zero-support and
the root profiles as one-class challenges.

The optional-field label-balance gate is **not applicable** to this benchmark
scope. It is not a failed corpus target and must not be pursued by generating
additional `BACKWARD` examples, because the executable oracle would simply
produce more SAFE labels.

## Future work, deliberately separate from this benchmark

1. If optional-field behaviour needs model scoring, first run fresh
   oracle-conformance under a separately declared `FORWARD` or `FULL` scope,
   or under an approved policy file that demonstrably changes the oracle
   outcome. Generate and audit a new corpus only if that conformance shows
   valid label variation. Do not mix those records with V9.
2. Test the trained model on independently sourced, real contract transitions
   before considering any operational use. Preserve the oracle as the final
   compatibility decision-maker.
3. For future generated benchmarks, run the near-duplicate audit before
   interpreting unusually clean held-out rows; report feature-space
   interpolation separately from broader structural extrapolation.

## Closeout status

V5 compositional policy semantics, V6 feature work, V7 failure filtering, V8
structural-profile gates, and the V9 benchmark/audit/three-seed experiment are
complete. The only optional-field limitation is documented as a qualified
oracle-invariant traceability case, not silently omitted or reported as a weak
model score.
