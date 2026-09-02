# V9 external evaluation through BACKWARD V10: final closeout

Status: **closed** on 2026-09-02. This document closes the Stripe,
Kubernetes, and Vega external-evaluation track. It authorizes no deployment,
new sourcing, calibration change, model change, or follow-on evaluation.

## What this track established—and what it did not

### Claim A — Pipeline and process verification: established

The track provides strong evidence that the evaluation process works as
designed under real adversarial conditions:

- the provenance and contamination gates rejected source/family/pair,
  exact-feature, and near-structural overlap rather than allowing convenient
  but contaminated examples into accuracy metrics;
- the corrected Vega provenance trail caught, archived, and replaced a flawed
  source reconstruction instead of silently reusing it;
- the runner enforces a distinct V9-training/BACKWARD-V10-execution identity
  boundary. The registered pairing is tied to an exhaustive 68,820-record
  behavioral audit, and negative controls demonstrate sensitivity to shared
  BACKWARD risks and resolver differences;
- the runner rejects missing or unaudited identities and specifically rejects
  any report that marks BACKWARD V10 as interchangeable with the unavailable
  V9 JAR; and
- sparse-evidence discipline held through final reporting. Seed repetitions
  were not counted as independent contracts, zero-support strata remained
  visible, and no sparse stratum was promoted as accuracy evidence.

This is a process-verification claim. It does not imply that V9 generalizes to
real contracts.

### Claim B — External generalization evidence for V9: not established

Only **two independent, non-contaminated public transitions** were scored.
Both came from Kubernetes, used the `baseline` policy pack, and were classified
as `CONSTRAINT_OR_RESTRICTION_CHANGED`. Repeating those two transitions across
three frozen seeds produced six inference invocations, not six independent
examples.

That support is explicitly insufficient for an accuracy or external-
generalization claim about V9. The aggregate and every observed stratum are
therefore `traceability-only`, with `accuracy_evidence=false` and
`reported_accuracy=null`. This evaluation **must not be cited in any future
summary as evidence that V9 is accurate on, transfers to, or is ready for
real-world contracts**.

## Actual purpose

This evaluation served ongoing model development and audit-quality process
verification. It tested provenance, isolation, oracle identity, inference,
and reporting mechanics. It was **not a deployment approval gate**, and this
closeout is **not a deployment sign-off**.

## Public-source yield

The three domains collectively exercised five mutation categories:
`ENUM_CHANGED`, `FIELD_ADDED`, `FIELD_REMOVED`,
`CONSTRAINT_OR_RESTRICTION_CHANGED`, and `TYPE_CHANGED`.

| Public source | Candidates found | Eligible after contamination/overlap gates | Final independent transitions scored | Disposition |
|---|---:|---:|---:|---|
| Stripe OpenAPI | 46 | 0 | 0 | 30 overlap-rejected; 16 explicitly traceability-only |
| Kubernetes OpenAPI | 10 | 2 | 2 | Seven `FIELD_REMOVED` and one constraint-change candidate rejected; two constraint changes accepted |
| Vega/Vega-Lite schemas | 2 | 0 | 0 | Both corrected `TYPE_CHANGED` candidates rejected at structural distance 1 |
| **Total** | **58** | **2** | **2** | Six seed invocations were produced from the two independent transitions |

The yield records diminishing returns from continued public-source hunting:
three independent domains and five mutation types produced only two eligible
transitions.

## Secondary unresolved observation—not a finding

The two independent scored transitions both drove the frozen models toward
`BREAKING`:

- Kubernetes `HTTPIngressPath`: oracle `BREAKING`, predicted `BREAKING` by all
  three seeds; and
- Kubernetes `PodFailurePolicyRule`: oracle `SAFE`, predicted `BREAKING` by all
  three seeds.

Thus the seed-level evidence contains three `BREAKING`→`BREAKING` invocations
and three `SAFE`→`BREAKING` invocations. This is **n=2 independent
transitions**, is not interpretable as a rate or model tendency, and is not an
accuracy finding. It remains an unresolved observation to watch only if
independent `CONSTRAINT_OR_RESTRICTION_CHANGED` coverage is obtained in a
separately authorized project.

## Backlog—not in progress

Named backlog item: **Manually authored, deliberately structurally-novel
external transitions**.

This is a future, separately scoped project, not a continuation of public-
repository hunting and not work currently in progress. Its purpose would be
to supply deliberately different structural cases that public histories did
not yield. Because manually authored records have no independent external
repository, commit history, or upstream blob identity to establish origin,
the project would need its own provenance discipline: pre-registered design
constraints, author and creation records, immutable content hashes, explicit
separation from V9 training generators and families, independent review, and
the same exact/near-structural contamination gates before any scoring claim.

## Self-contained evidence and hash ledger

All hashes are SHA-256 unless identified as historical Git blob SHA-1 values
inside the source manifests. BACKWARD V10 remains a distinct identity and is
never interchangeable with the unavailable V9 executable.

### Frozen identities and equivalence boundary

| Artifact or identity | SHA-256 |
|---|---|
| [V9 prepared training dataset](../data/generated/dcg-oracle-production-v9-structural-balanced.json) | `fa09e645480737ba856940778264284d8e832f4532e989c92a259320cfe05ad3` |
| Historical V9 training-oracle identity; original binary unavailable | `809b25e627e43f847f00a0ce87dcde33ad359006bf22be403e26d662274677dc` |
| [BACKWARD V10 execution JAR](../data/oracle-binaries/backward-v10/contract-cli-c00da951bac88d2b.jar) | `c00da951bac88d2be245e08917e6fc62f55abf3a50aad2dea67178a82285f6b6` |
| [Pinned policy packs](../data-contract-governance/contracts/policy-packs-v5-compositional.json) | `8f82b058f81ace43c89180803c7ec26ac734b84d0092036a77115688337e1bb6` |
| [Behavioral-equivalence audit](../data/oracle-binaries/backward-v10/behavioral-equivalence-audit-v1.json) | `c090f38c178d5652b788501ec7a26919ec24bd215a32d344dbeb2d9392634793` |
| [Reconstruction checkpoint](../data/oracle-binaries/backward-v10/reconstruction-checkpoint-v1.json) | `aa6b12250edea082447c1b9532bb7d1965e89ddf297862835742611055013f08` |
| [Reproducible-build overlay](../data/oracle-binaries/backward-v10/reproducible-build-overlay-v1.patch) | `9485abef0644a7fbf750d24b5a353ff748ce66e9b4b55d041a5f51799e30b8f0` |

The dual-provenance enforcement and its tests are in
[`src/models/external_evaluation.rs`](../src/models/external_evaluation.rs),
SHA-256
`c5650c0a5441ddd4cf22a57b286fb878107629b105c98fb9ba97163839e71696`.
The test
`scored_report_rejects_distinct_oracles_marked_interchangeable` sets
`identity_interchangeable=true` while retaining the approved V9→V10 hashes and
asserts the exact rejection returned by `require_scored_dual_provenance()`.
Companion tests reject missing training/execution hashes and unaudited oracle
pairings.

### Source provenance and contamination evidence

| Artifact | SHA-256 |
|---|---|
| [Stripe transition manifest](../data/external/stripe-openapi-v3/stripe-openapi-external-manifest-v3.json) | `be66872fcb14034807faad297b81b1a932bc98e87e4a8538e9615d1a40a522ce` |
| [Stripe original preflight](../data/external/stripe-openapi-v3/v9-preflight-report.json) | `48cf97dfad2e9481472e27b7ff890fd69312d0cb0cde51f4b9b11266eec43e28` |
| [Stripe final preflight](../data/external/stripe-openapi-v3/v9-preflight-report-after-traceability.json) | `bb2795538e6e17f586b6dd27e9917ece03b99a0c02195368bb7a3ef57075fc14` |
| [Stripe traceability-only audit](../data/external/stripe-openapi-v3/stripe-traceability-only-audit.json) | `d3b394b4f9923d419cff6e0625bc18b9ceac01067ec9889b60d6e44642e0bbdb` |
| [Kubernetes removal-biased prescreen and projection](../data/external/kubernetes-openapi-v1/kubernetes-removal-biased-prescreen-v1.json) | `699d8ed09d3eeebd9f2676ad6da98bbd68d32548310e04770744dc9aeeb3229c` |
| [Vega corrected prescreen](../data/external/vega-schema-type-change-v1/vega-type-change-prescreen-v2.json) | `6b3edf9667716fe6898913c776a60ca43230fbb5282774153d0a9512ec91ba97` |
| [Vega corrected candidate manifest](../data/external/vega-schema-type-change-v1/vega-type-change-candidates-v4.json) | `3fc4791f480c0987f376b98954ae34f6fb91f19ec76c5d5874c6ca4213888fea` |
| [Vega corrected overlap projection](../data/external/vega-schema-type-change-v1/vega-type-change-overlap-projection-v2.json) | `d35013e34750828f0de060ac059e360367c04e3596ce60a5e93f09127faee250` |
| [Vega canonical provenance freeze](../data/external/vega-schema-type-change-v1/canonical-provenance-freeze-v1.json) | `db5e04adf3eeb356a4c888e34f70a02f8fafbdcfbf0b0f7d5a1dc851d2a8e1e1` |
| [Vega provenance correction](../data/external/vega-schema-type-change-v1/provenance-integrity-correction-v2.json) | `fcdb2a35623d60a7bc6131196b437a2f9a5a16a737cb5b8a478ebea6e1999ed1` |
| [Vega ConditionalPredicate diagnostic](../data/external/vega-schema-type-change-v1/diagnostics-v2/conditional-predicate-feature-diagnostic-v2.json) | `14bc235e50371b0354644ed772ae60f78717ca98a72dafa15daab15b0fddd003` |
| [Vega ConditionalSelection diagnostic](../data/external/vega-schema-type-change-v1/diagnostics-v2/conditional-selection-feature-diagnostic-v2.json) | `6f71b11fec469a1d12e5e1d5e601302e44c4d7c16393a0dae4d3d1abf024da34` |
| [Vega ConditionalPredicate trace](../data/external/vega-schema-type-change-v1/traces-v2/conditional-predicate-raw-trace-v2.json) | `950799efe52b550db72840d6cfafe153f76a6524ca52651d8a63f375efa02bd7` |
| [Vega ConditionalSelection trace](../data/external/vega-schema-type-change-v1/traces-v2/conditional-selection-raw-trace-v2.json) | `a3181d9b818228ff916e65e6f358b1a18404dc0e1585446d612fb1c5decd6171` |
| [Root-removal saturation evidence](../data/external/field-removed-root-pattern-saturation-v1.json) | `389ea5547e30d86bbfcedad8f6b65db08dc1c5ecd38c1ce1d2fe6cf625f5e3dd` |
| [Stripe root-removal diagnostic](../data/external/stripe-openapi-v3/field-removed-card-feature-diagnostic-v1.json) | `baddec5d7b8f2edcadb0160ecfb682ab70bc38b2b03403a4d9a1059177ff2882` |
| [Kubernetes root-removal diagnostic](../data/external/kubernetes-openapi-v1/field-removed-object-meta-feature-diagnostic-v1.json) | `3d3022218f956138db15aaf1835be66fa81ae028d39261b366eca0d6b538547c` |

The superseded Vega v1 artifacts remain only under
[`provenance-bug-archive-v1`](../data/external/vega-schema-type-change-v1/provenance-bug-archive-v1/);
their frozen hashes are recorded in the canonical provenance freeze and they
were not used by scoring.

### Scored Kubernetes input and validation history

The final input contains only the two rows already accepted by the persisted
Kubernetes projection. The exact upstream source files and their Git blob
identities are recorded in the manifest; symmetric reference closure made the
two component projections validator-readable without changing their target
required-array differences or distance-2 contamination result.

| Artifact | SHA-256 |
|---|---|
| [Kubernetes v1.21.0 source](../data/external/kubernetes-openapi-v1/scored-accepted-v1/sources/v1.21.0-swagger.json) | `128a984dbb5a4e5ceceef9dea0db575267678d333f53ed606300a2132d2539cc` |
| [Kubernetes v1.22.0 source](../data/external/kubernetes-openapi-v1/scored-accepted-v1/sources/v1.22.0-swagger.json) | `d6718670e062681e4dc9e2b9dadf2b311c147cfdabcac628b9018011165d8773` |
| [Kubernetes v1.28.0 source](../data/external/kubernetes-openapi-v1/scored-accepted-v1/sources/v1.28.0-swagger.json) | `c3d62644ad31aeca618e6f640178caf39f917ba296b4919443d9838a22a10be1` |
| [Kubernetes v1.29.0 source](../data/external/kubernetes-openapi-v1/scored-accepted-v1/sources/v1.29.0-swagger.json) | `84c20b1e2bcedce1d9d09c03e0ccb02500cd96e62e4fa54651b0e97b3a211046` |
| [Component-only manifest v1](../data/external/kubernetes-openapi-v1/scored-accepted-v1/kubernetes-accepted-external-manifest-v1.json) | `53b5ece454472a5907149ae17c93adc7dfc7c7b2572ea6aeb4cca82d3c270ee8` |
| [V1 preflight](../data/external/kubernetes-openapi-v1/scored-accepted-v1/preflight-backward-v10-v1.json) | `9f1b2591f804fe041f0672173698c4e85a0f1a90af8f38c9a62247e6ae59abf2` |
| [Reference-closure manifest v2](../data/external/kubernetes-openapi-v1/scored-accepted-v1/kubernetes-accepted-external-manifest-v2.json) | `a67e8eb1133cf453f63413c61be849388b5bce573c813f1dc8aa3dcfe64d2051` |
| [V2 preflight](../data/external/kubernetes-openapi-v1/scored-accepted-v1/preflight-backward-v10-v2.json) | `6446974c8b61d6b003f5fb797b2248b3212402198d6461f227bc067714d618f6` |
| [Final type-bearing reference-closure manifest v3](../data/external/kubernetes-openapi-v1/scored-accepted-v1/kubernetes-accepted-external-manifest-v3.json) | `cf0eafd53c47d9000b020063273e6327d7d8b783d1f51606c1dce68f1f8e195e` |
| [Final V3 preflight, 2/2 accepted](../data/external/kubernetes-openapi-v1/scored-accepted-v1/preflight-backward-v10-v3.json) | `3b47d7eb6e846308beaeb00d0fe448ed2b31c9a2b0a53571a320a519ecd90bcd` |

The v1 and v2 seed reports record oracle validation rejection before
inference. They are audit history, not scored evidence:

| Validation-rejection report | SHA-256 |
|---|---|
| [V1 seed 20260826](../data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/seed-20260826-report.json) | `712573ad33fb86ae3ac6738b5d5db6a062b91c3107f2755e28f0c8d6cec378cd` |
| [V1 seed 20260827](../data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/seed-20260827-report.json) | `7e52dd100ad9591eab0a36e8348262d37518e75982a7f8f262747a9fbe18d7cb` |
| [V1 seed 20260828](../data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/seed-20260828-report.json) | `fa5d59f60552b9f3e6f8313768c8dcc6b823dbf8c76245bb2bbbdbf923bb1ee3` |
| [V2 seed 20260826](../data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/seed-20260826-report-v2.json) | `2d825cc3aaabff141efc903a477e5f220b71b11da268487987b91c38e64edebc` |
| [V2 seed 20260827](../data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/seed-20260827-report-v2.json) | `52b847c3ed0dad7acc4a3315fc94e89ec45814661b6313707d86f6f3e5550488` |
| [V2 seed 20260828](../data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/seed-20260828-report-v2.json) | `73c8e1221fbe92aec17983c7ca62db5cb135f41778e8a842d83176dbec2cc4ad` |

Only the v3 reports below contain predictions.

### Frozen models, scored reports, and final synthesis

| Artifact | SHA-256 |
|---|---|
| [Seed 20260826 model](../data/experiments/v9-multiclass-cpu-100e/models/seed-20260826-normal-family-split.json) | `5da2fedbee5d1b3c84c79cb75e2cd10c0b3462066b66571570e93fb7ccd84988` |
| [Seed 20260827 model](../data/experiments/v9-multiclass-cpu-100e/models/seed-20260827-normal-family-split.json) | `bd464322c272b8ec1d5ab88605022d4a48e3738b63ab1791a4c7e56f37222ac8` |
| [Seed 20260828 model](../data/experiments/v9-multiclass-cpu-100e/models/seed-20260828-normal-family-split.json) | `24ec9e4e758370093c228af34e3b05ac65820b3fd413ca80a269206ce1edd2c0` |
| [Seed 20260826 scored report](../data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/seed-20260826-report-v3.json) | `c907642d832e855d6f720f49ccd759ef237c370e9fbdd483336579e549460eb0` |
| [Seed 20260827 scored report](../data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/seed-20260827-report-v3.json) | `9b3a9d84667e710cc64d4aeb5fb9f9e28fe139c0781d69cad921f53c558183dd` |
| [Seed 20260828 scored report](../data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/seed-20260828-report-v3.json) | `9548e0bcee475eb5b0525acfae2734529980edb997868017aab4ce558e0ca832` |
| [Three-seed aggregate and invocation traces](../data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/three-seed-aggregate-report-v1.json) | `a05b7cd8b5ba8c70134466988f90ffd65be790cbc54a27a5e34a2ea56e08d413` |
| [Scoring-run artifact hash inventory](../data/experiments/external-backward-v10/stripe-kubernetes-vega-v1/SHA256SUMS-v1.txt) | `b935c59f56e1588b0d14e2e6c852f4453cca2b53adfbacddd93c498bc9a5f766` |

Together with the aggregate's invocation-level traces, that inventory is the
canonical machine-readable index for the scoring run. The broader source and
provenance hashes are recorded directly in this closeout ledger.
