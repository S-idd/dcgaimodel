## Summary

Describe the change and why it is needed.

## Verification

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --all-targets --all-features --locked -- -D warnings`
- [ ] `cargo test --all-targets --all-features --locked`
- [ ] Documentation and changelog updated when applicable

## Safety and provenance

- [ ] Model behavior remains advisory and cannot override deterministic enforcement
- [ ] Dataset, family-isolation, and provenance implications are documented
- [ ] No secrets, generated corpora, oracle staging files, or external repository checkouts are included
