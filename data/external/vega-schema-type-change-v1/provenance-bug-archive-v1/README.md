# Superseded Vega provenance artifacts

These artifacts are retained only to audit a provenance-integrity correction.
They must not be used as current evidence.

The original pre-screen correctly preserved and diffed the two versioned schema
blobs, but incorrectly copied the common repository snapshot commit into both
per-version commit fields. A first attempted correction used a shallow source
clone and was also rejected because it could not resolve complete file history.

Use the V2 pre-screen, V4 candidate manifest, V2 projection, and V2 traces in
the parent directory. The machine-readable correction is
`../provenance-integrity-correction-v2.json`.
