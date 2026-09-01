# Contributing to dcgaimodel

## Development Philosophy

Thank you for helping improve `dcgaimodel`. This project combines a Rust learning framework with
reproducible Data Contract Governance research, so correctness, provenance, and clear safety
boundaries matter.

Every feature should be:

- Well documented
- Unit tested
- Idiomatic Rust
- Modular
- Easy to understand

## Development workflow

1. Open an issue for substantial behavior or data-contract changes.
2. Create a focused branch from the current development branch.
3. Add or update tests with the implementation.
4. Run the local quality gate:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --all-targets --all-features --locked -- -D warnings
   cargo test --all-targets --all-features --locked
   ```

5. Update documentation and `CHANGELOG.md` when behavior or artifact formats change.
6. Open a pull request using the repository template.

## Commit style

Examples:

```
Implement vector addition

Implement matrix multiplication

Add SGD optimizer

Fix dot product overflow
```

## Coding standards

- Avoid duplicated code.
- Prefer composition over duplication.
- Document public APIs.
- Handle expected failures using `Result`.
- Never panic unless absolutely necessary.
- Keep modules focused on a single responsibility.
- Preserve deterministic seeds, family isolation, and provenance hashes in evaluation work.
- Keep model output advisory; it must not override deterministic compatibility enforcement.

## Testing

Every public function should have at least one unit test.

New functionality must include tests before merging. Tests that require an external JAR must remain
explicitly opt-in and document the required pinned artifact.

## Generated and external artifacts

Do not commit Cargo build output, oracle staging directories, generated corpora, external repository
checkouts, or secrets. Small, reviewed evidence and manifests may be committed under `data/` when
their provenance and purpose are documented.

The deterministic governance engine lives in the separate
[`S-idd/data-contract-governance`](https://github.com/S-idd/data-contract-governance) repository.
Clone it beside the paths documented in `docs/` only when an opt-in oracle workflow requires it.

## Community standards

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Report vulnerabilities using
the private process in [SECURITY.md](SECURITY.md).
