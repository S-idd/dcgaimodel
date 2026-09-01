# Prepared dataset format

`dcg-prepared-dataset-v2` is a portable JSON representation for validated training records. It contains a dataset version and one exact feature-schema version (`dcg-features-v1` or `dcg-features-v2`) plus every record's:

- record and source IDs;
- `family_id` and `split_group_id`;
- `dataset_role` (`standard` or a whole-family `challenge` reservation);
- contract ID, old/new version, and resolved policy pack;
- raw validated feature vector; and
- binary targets for breaking change and incompatibility plus the advisory-risk target; and
- when oracle-generated, the canonical `safe` / `warning` / `breaking` outcome and pinned-oracle provenance.

No absolute filesystem paths, credentials, or Java-runtime directories are recorded. Loading validates all metadata, feature values/order, and labels.

`PreparedDcgDataset::split_by_group` deterministically assigns complete standard `split_group_id`s to train, validation, or test. Policy-pack variants and contract-family siblings therefore remain together; challenge families are never included in this split. It requires at least three independent standard groups and uses the configured seed only for stable group ordering. Duplicate record IDs, invalid feature vectors, invalid canonical labels, and incompatible feature versions are rejected while loading.

For generated records, provenance also retains the pinned-JAR SHA-256, the
oracle result, mutation proposal, direction, generator version/seed, and a
canonical pair fingerprint. `generalization_readiness()` derives reusable
shortcut and challenge-protocol reports from these persisted fields.
