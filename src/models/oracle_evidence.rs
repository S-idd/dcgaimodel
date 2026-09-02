//! Portable, qualified evidence from the executable DCG compatibility oracle.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub const ORACLE_CONFORMANCE_FORMAT_VERSION: &str = "dcg-oracle-conformance-v2";
const LEGACY_ORACLE_CONFORMANCE_FORMAT_VERSION: &str = "dcg-oracle-conformance-v1";
pub const ORACLE_PROMOTION_FORMAT_VERSION: &str = "dcg-oracle-invariant-promotions-v1";
/// Exact identity of the mechanism-only FIELD_REMOVED severity fixture. Any
/// production corpus or benchmark path must reject this identity.
pub const FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_SHA256: &str =
    "33122df6e7b584935918567d683e8a314afdd53db4499f29791f94fbe9d6fcf6";
pub const FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_FILE_NAME: &str =
    "field-removed-severity-audit-fixture-v1.json";

/// Compatibility direction passed verbatim to the pinned CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OracleCompatibilityMode {
    Backward,
    Forward,
    Full,
}

impl OracleCompatibilityMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Backward => "BACKWARD",
            Self::Forward => "FORWARD",
            Self::Full => "FULL",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "BACKWARD" => Some(Self::Backward),
            "FORWARD" => Some(Self::Forward),
            "FULL" => Some(Self::Full),
            _ => None,
        }
    }

    pub const fn all() -> [Self; 3] {
        [Self::Backward, Self::Forward, Self::Full]
    }
}

/// Configurable governance thresholds for a candidate invariant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InvariantEvidenceConfig {
    pub min_independent_families: usize,
    pub min_policies: usize,
    pub min_oracle_checks: usize,
}

impl Default for InvariantEvidenceConfig {
    fn default() -> Self {
        Self {
            min_independent_families: 25,
            min_policies: 3,
            min_oracle_checks: 75,
        }
    }
}

/// One unmodified captured invocation from the pinned executable JAR.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleConformanceRun {
    pub family_id: String,
    pub mutation_family: String,
    #[serde(default)]
    pub mutation_variant: String,
    #[serde(default)]
    pub consumer_profile: String,
    pub compatibility_mode: OracleCompatibilityMode,
    pub policy_pack: String,
    pub exit_code: i32,
    pub outcome: String,
    pub stdout: String,
    pub stderr: String,
    #[serde(default)]
    pub rejection_stage: Option<String>,
    #[serde(default)]
    pub rejection_reason: Option<String>,
}

/// One schema-lint preflight performed before a candidate enters the policy
/// and direction matrix.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleConformancePreflight {
    pub family_id: String,
    pub mutation_family: String,
    pub mutation_variant: String,
    pub consumer_profile: String,
    pub accepted: bool,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub rejection_stage: Option<String>,
    pub rejection_reason: Option<String>,
}

/// Matrix cell evidence, qualified to exactly one mutation, mode, and policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleConformanceCell {
    pub mutation_family: String,
    pub compatibility_mode: OracleCompatibilityMode,
    pub policy_pack: String,
    pub independent_families: usize,
    pub oracle_checks: usize,
    /// Accepted (exit-0/1) checks; exit-2 calls remain in the raw matrix but
    /// cannot establish a valid transition invariant.
    #[serde(default)]
    pub accepted_oracle_checks: usize,
    /// Labels from accepted (exit-0/1) oracle calls only. Rejected exit-2
    /// invocations remain represented in `labels_observed` and raw `runs`.
    #[serde(default)]
    pub accepted_labels_observed: BTreeMap<String, usize>,
    pub labels_observed: BTreeMap<String, usize>,
    pub candidate_invariant: bool,
}

/// Cross-policy evidence for one mutation and one compatibility mode.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleCrossPolicyEvidence {
    pub mutation_family: String,
    pub compatibility_mode: OracleCompatibilityMode,
    pub independent_families: usize,
    pub policies_checked: Vec<String>,
    /// Policies with at least one accepted oracle result. A policy whose only
    /// calls are rejected cannot establish an invariant.
    #[serde(default)]
    pub accepted_policies_checked: Vec<String>,
    pub oracle_checks: usize,
    #[serde(default)]
    pub accepted_oracle_checks: usize,
    /// Labels from accepted (exit-0/1) oracle calls only. Raw rejected calls
    /// remain available in `labels_observed` and the complete run list.
    #[serde(default)]
    pub accepted_labels_observed: BTreeMap<String, usize>,
    pub labels_observed: BTreeMap<String, usize>,
    pub candidate_invariant: bool,
}

/// Saved audit artifact. Raw runs preserve every requested mode/policy outcome.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleConformanceReport {
    pub format_version: String,
    pub oracle_jar_sha256: String,
    pub policy_packs_sha256: String,
    pub generator_version: String,
    pub thresholds: InvariantEvidenceConfig,
    #[serde(default)]
    pub preflight_runs: Vec<OracleConformancePreflight>,
    pub runs: Vec<OracleConformanceRun>,
    pub cells: Vec<OracleConformanceCell>,
    pub cross_policy: Vec<OracleCrossPolicyEvidence>,
}

/// An explicit, reviewer-approved exception. Its identity fields make it stale
/// automatically if the JAR, policy-pack file, mode, or policy changes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualifiedOracleInvariantPromotion {
    pub mutation_family: String,
    pub compatibility_mode: OracleCompatibilityMode,
    pub policy_pack: String,
    pub observed_label: String,
    pub oracle_jar_sha256: String,
    pub policy_packs_sha256: String,
    pub independent_families: usize,
    pub oracle_checks: usize,
    pub rationale: String,
}

/// Explicit promotion manifest consumed by benchmark readiness. Candidate
/// discovery never mutates this artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleInvariantPromotionManifest {
    pub format_version: String,
    pub promotions: Vec<QualifiedOracleInvariantPromotion>,
}

impl OracleConformanceReport {
    pub fn new(
        oracle_jar_sha256: String,
        policy_packs_sha256: String,
        thresholds: InvariantEvidenceConfig,
        runs: Vec<OracleConformanceRun>,
    ) -> Self {
        Self::new_with_preflights(
            oracle_jar_sha256,
            policy_packs_sha256,
            thresholds,
            Vec::new(),
            runs,
        )
    }

    pub fn new_with_preflights(
        oracle_jar_sha256: String,
        policy_packs_sha256: String,
        thresholds: InvariantEvidenceConfig,
        mut preflight_runs: Vec<OracleConformancePreflight>,
        mut runs: Vec<OracleConformanceRun>,
    ) -> Self {
        preflight_runs.sort_by(|left, right| {
            (
                &left.family_id,
                &left.consumer_profile,
                &left.mutation_family,
                &left.mutation_variant,
            )
                .cmp(&(
                    &right.family_id,
                    &right.consumer_profile,
                    &right.mutation_family,
                    &right.mutation_variant,
                ))
        });
        runs.sort_by(|left, right| {
            (
                &left.mutation_family,
                &left.mutation_variant,
                &left.consumer_profile,
                left.compatibility_mode,
                &left.policy_pack,
                &left.family_id,
            )
                .cmp(&(
                    &right.mutation_family,
                    &right.mutation_variant,
                    &right.consumer_profile,
                    right.compatibility_mode,
                    &right.policy_pack,
                    &right.family_id,
                ))
        });
        let cells = aggregate_cells(&runs, &thresholds);
        let cross_policy = aggregate_cross_policy(&runs, &thresholds);
        Self {
            format_version: ORACLE_CONFORMANCE_FORMAT_VERSION.to_owned(),
            oracle_jar_sha256,
            policy_packs_sha256,
            generator_version: env!("CARGO_PKG_VERSION").to_owned(),
            thresholds,
            preflight_runs,
            runs,
            cells,
            cross_policy,
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let bytes = serde_json::to_vec_pretty(self).map_err(|error| error.to_string())?;
        fs::write(path, bytes).map_err(|error| error.to_string())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let mut report: Self =
            serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        if report.format_version != ORACLE_CONFORMANCE_FORMAT_VERSION
            && report.format_version != LEGACY_ORACLE_CONFORMANCE_FORMAT_VERSION
        {
            return Err(format!(
                "unsupported conformance report: {}",
                report.format_version
            ));
        }
        report.format_version = ORACLE_CONFORMANCE_FORMAT_VERSION.to_owned();
        report.rebuild_aggregates();
        Ok(report)
    }

    /// Rebuilds derived candidate evidence from immutable raw oracle runs.
    /// This supports upgraded qualification logic without ever changing the
    /// captured invocation evidence, JAR identity, or policy-pack identity.
    pub fn rebuild_aggregates(&mut self) {
        self.cells = aggregate_cells(&self.runs, &self.thresholds);
        self.cross_policy = aggregate_cross_policy(&self.runs, &self.thresholds);
    }

    /// Creates promotions only for explicit reviewer selections that remain
    /// candidates in both the policy-specific and cross-policy evidence.
    pub fn promote(
        &self,
        selections: &[(String, OracleCompatibilityMode, String, String)],
    ) -> Result<OracleInvariantPromotionManifest, String> {
        let mut promotions = Vec::new();
        for (mutation, mode, policy, rationale) in selections {
            if rationale.trim().is_empty() {
                return Err(format!("promotion rationale is empty for {mutation}"));
            }
            let cell = self
                .cells
                .iter()
                .find(|cell| {
                    cell.mutation_family == *mutation
                        && cell.compatibility_mode == *mode
                        && cell.policy_pack == *policy
                })
                .ok_or_else(|| {
                    format!(
                        "missing conformance cell for {mutation}/{}/{policy}",
                        mode.as_str()
                    )
                })?;
            let cross = self
                .cross_policy
                .iter()
                .find(|cross| {
                    cross.mutation_family == *mutation && cross.compatibility_mode == *mode
                })
                .ok_or_else(|| {
                    format!(
                        "missing cross-policy evidence for {mutation}/{}",
                        mode.as_str()
                    )
                })?;
            if !cell.candidate_invariant || !cross.candidate_invariant {
                return Err(format!(
                    "{mutation}/{}/{policy} is not a promotable candidate",
                    mode.as_str()
                ));
            }
            let observed_label = single_label(&cell.accepted_labels_observed)
                .ok_or_else(|| format!("candidate {mutation} unexpectedly has mixed labels"))?;
            promotions.push(QualifiedOracleInvariantPromotion {
                mutation_family: mutation.clone(),
                compatibility_mode: *mode,
                policy_pack: policy.clone(),
                observed_label,
                oracle_jar_sha256: self.oracle_jar_sha256.clone(),
                policy_packs_sha256: self.policy_packs_sha256.clone(),
                independent_families: cell.independent_families,
                oracle_checks: cell.oracle_checks,
                rationale: rationale.clone(),
            });
        }
        promotions.sort_by(|left, right| {
            (
                &left.mutation_family,
                left.compatibility_mode,
                &left.policy_pack,
            )
                .cmp(&(
                    &right.mutation_family,
                    right.compatibility_mode,
                    &right.policy_pack,
                ))
        });
        Ok(OracleInvariantPromotionManifest {
            format_version: ORACLE_PROMOTION_FORMAT_VERSION.to_owned(),
            promotions,
        })
    }
}

impl OracleInvariantPromotionManifest {
    pub fn empty() -> Self {
        Self {
            format_version: ORACLE_PROMOTION_FORMAT_VERSION.to_owned(),
            promotions: Vec::new(),
        }
    }
    pub fn save(&self, path: &Path) -> Result<(), String> {
        fs::write(
            path,
            serde_json::to_vec_pretty(self).map_err(|error| error.to_string())?,
        )
        .map_err(|error| error.to_string())
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let manifest: Self =
            serde_json::from_slice(&fs::read(path).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
        if manifest.format_version != ORACLE_PROMOTION_FORMAT_VERSION {
            return Err(format!(
                "unsupported promotion manifest: {}",
                manifest.format_version
            ));
        }
        Ok(manifest)
    }

    pub fn matching(
        &self,
        mutation: &str,
        mode: OracleCompatibilityMode,
        policy: &str,
        label: &str,
        jar_hash: &str,
        policy_hash: &str,
    ) -> Option<&QualifiedOracleInvariantPromotion> {
        self.promotions.iter().find(|promotion| {
            promotion.mutation_family == mutation
                && promotion.compatibility_mode == mode
                && promotion.policy_pack == policy
                && promotion.observed_label == label
                && promotion.oracle_jar_sha256 == jar_hash
                && promotion.policy_packs_sha256 == policy_hash
        })
    }

    /// Combines independently reviewed evidence without allowing an ambiguous
    /// mutation/mode/policy scope to be promoted twice.
    pub fn merge(mut self, additions: Self) -> Result<Self, String> {
        for promotion in additions.promotions {
            if self.promotions.iter().any(|existing| {
                existing.mutation_family == promotion.mutation_family
                    && existing.compatibility_mode == promotion.compatibility_mode
                    && existing.policy_pack == promotion.policy_pack
            }) {
                return Err(format!(
                    "duplicate promotion scope: {}/{}/{}",
                    promotion.mutation_family,
                    promotion.compatibility_mode.as_str(),
                    promotion.policy_pack
                ));
            }
            self.promotions.push(promotion);
        }
        self.promotions.sort_by(|left, right| {
            (
                &left.mutation_family,
                left.compatibility_mode,
                &left.policy_pack,
            )
                .cmp(&(
                    &right.mutation_family,
                    right.compatibility_mode,
                    &right.policy_pack,
                ))
        });
        Ok(self)
    }
}

fn aggregate_cells(
    runs: &[OracleConformanceRun],
    thresholds: &InvariantEvidenceConfig,
) -> Vec<OracleConformanceCell> {
    let mut groups =
        BTreeMap::<(String, OracleCompatibilityMode, String), Vec<&OracleConformanceRun>>::new();
    for run in runs {
        groups
            .entry((
                run.mutation_family.clone(),
                run.compatibility_mode,
                run.policy_pack.clone(),
            ))
            .or_default()
            .push(run);
    }
    let min_cell_checks = thresholds
        .min_oracle_checks
        .div_ceil(thresholds.min_policies);
    groups
        .into_iter()
        .map(
            |((mutation_family, compatibility_mode, policy_pack), runs)| {
                let accepted_runs = runs
                    .iter()
                    .copied()
                    .filter(|run| run.outcome != "rejected")
                    .collect::<Vec<_>>();
                let families = accepted_runs
                    .iter()
                    .map(|run| &run.family_id)
                    .collect::<BTreeSet<_>>()
                    .len();
                let labels = label_counts(&runs);
                let accepted_labels = label_counts(&accepted_runs);
                let checks = runs.len();
                let accepted_checks = accepted_runs.len();
                OracleConformanceCell {
                    mutation_family,
                    compatibility_mode,
                    policy_pack,
                    independent_families: families,
                    oracle_checks: checks,
                    accepted_oracle_checks: accepted_checks,
                    accepted_labels_observed: accepted_labels.clone(),
                    candidate_invariant: families >= thresholds.min_independent_families
                        && accepted_checks >= min_cell_checks
                        && accepted_labels.len() == 1,
                    labels_observed: labels,
                }
            },
        )
        .collect()
}

fn aggregate_cross_policy(
    runs: &[OracleConformanceRun],
    thresholds: &InvariantEvidenceConfig,
) -> Vec<OracleCrossPolicyEvidence> {
    let mut groups =
        BTreeMap::<(String, OracleCompatibilityMode), Vec<&OracleConformanceRun>>::new();
    for run in runs {
        groups
            .entry((run.mutation_family.clone(), run.compatibility_mode))
            .or_default()
            .push(run);
    }
    groups
        .into_iter()
        .map(|((mutation_family, compatibility_mode), runs)| {
            let accepted_runs = runs
                .iter()
                .copied()
                .filter(|run| run.outcome != "rejected")
                .collect::<Vec<_>>();
            let families = accepted_runs
                .iter()
                .map(|run| &run.family_id)
                .collect::<BTreeSet<_>>()
                .len();
            let policies = runs
                .iter()
                .map(|run| run.policy_pack.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let accepted_policies = accepted_runs
                .iter()
                .map(|run| run.policy_pack.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let labels = label_counts(&runs);
            let accepted_labels = label_counts(&accepted_runs);
            let checks = runs.len();
            let accepted_checks = accepted_runs.len();
            OracleCrossPolicyEvidence {
                mutation_family,
                compatibility_mode,
                independent_families: families,
                policies_checked: policies.clone(),
                accepted_policies_checked: accepted_policies.clone(),
                oracle_checks: checks,
                accepted_oracle_checks: accepted_checks,
                accepted_labels_observed: accepted_labels.clone(),
                candidate_invariant: families >= thresholds.min_independent_families
                    && accepted_policies.len() >= thresholds.min_policies
                    && accepted_checks >= thresholds.min_oracle_checks
                    && accepted_labels.len() == 1,
                labels_observed: labels,
            }
        })
        .collect()
}

fn label_counts(runs: &[&OracleConformanceRun]) -> BTreeMap<String, usize> {
    let mut labels = BTreeMap::new();
    for run in runs {
        *labels.entry(run.outcome.clone()).or_default() += 1;
    }
    labels
}
fn single_label(labels: &BTreeMap<String, usize>) -> Option<String> {
    (labels.len() == 1)
        .then(|| labels.keys().next().cloned())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn run(
        family: &str,
        mutation: &str,
        mode: OracleCompatibilityMode,
        policy: &str,
        outcome: &str,
    ) -> OracleConformanceRun {
        OracleConformanceRun {
            family_id: family.to_owned(),
            mutation_family: mutation.to_owned(),
            mutation_variant: String::new(),
            consumer_profile: String::new(),
            compatibility_mode: mode,
            policy_pack: policy.to_owned(),
            exit_code: match outcome {
                "breaking" => 1,
                "rejected" => 2,
                _ => 0,
            },
            outcome: outcome.to_owned(),
            stdout: String::new(),
            stderr: String::new(),
            rejection_stage: (outcome == "rejected").then(|| "compatibility_evaluation".to_owned()),
            rejection_reason: (outcome == "rejected").then(|| "test rejection".to_owned()),
        }
    }
    fn report(runs: Vec<OracleConformanceRun>) -> OracleConformanceReport {
        OracleConformanceReport::new(
            "jar".to_owned(),
            "packs".to_owned(),
            InvariantEvidenceConfig {
                min_independent_families: 2,
                min_policies: 3,
                min_oracle_checks: 6,
            },
            runs,
        )
    }
    #[test]
    fn candidate_requires_independent_families_policies_checks_and_one_label() {
        let mut runs = Vec::new();
        for family in ["a", "b"] {
            for policy in ["baseline", "strict", "relaxed"] {
                runs.push(run(
                    family,
                    "field_removed",
                    OracleCompatibilityMode::Backward,
                    policy,
                    "breaking",
                ));
            }
        }
        let report = report(runs);
        assert!(report.cross_policy[0].candidate_invariant);
        assert!(report.cells.iter().all(|cell| cell.candidate_invariant));
    }

    #[test]
    fn rejected_calls_are_audited_but_do_not_mix_accepted_evidence() {
        let mut runs = Vec::new();
        for family in ["a", "b"] {
            for policy in ["baseline", "strict", "relaxed"] {
                runs.push(run(
                    family,
                    "field_removed",
                    OracleCompatibilityMode::Backward,
                    policy,
                    "breaking",
                ));
            }
        }
        runs.push(run(
            "unsupported",
            "field_removed",
            OracleCompatibilityMode::Backward,
            "baseline",
            "rejected",
        ));
        let report = report(runs);
        assert!(report.cross_policy[0].candidate_invariant);
        assert_eq!(
            report.cross_policy[0].accepted_labels_observed,
            BTreeMap::from([("breaking".to_owned(), 6)])
        );
        assert_eq!(
            report.cross_policy[0]
                .labels_observed
                .get("rejected")
                .copied(),
            Some(1)
        );
    }
    #[test]
    fn repeated_rows_one_family_and_mixed_labels_do_not_promote() {
        let runs = vec![
            run(
                "one",
                "field_removed",
                OracleCompatibilityMode::Backward,
                "baseline",
                "breaking",
            ),
            run(
                "one",
                "field_removed",
                OracleCompatibilityMode::Backward,
                "strict",
                "breaking",
            ),
            run(
                "one",
                "field_removed",
                OracleCompatibilityMode::Backward,
                "relaxed",
                "warning",
            ),
        ];
        let report = report(runs);
        assert!(!report.cross_policy[0].candidate_invariant);
    }

    #[test]
    fn insufficient_policy_coverage_is_not_a_candidate() {
        let runs = vec![
            run(
                "a",
                "field_removed",
                OracleCompatibilityMode::Backward,
                "baseline",
                "breaking",
            ),
            run(
                "b",
                "field_removed",
                OracleCompatibilityMode::Backward,
                "baseline",
                "breaking",
            ),
        ];
        assert!(!report(runs).cross_policy[0].candidate_invariant);
    }
    #[test]
    fn promotion_is_explicit_and_identity_qualified() {
        let mut runs = Vec::new();
        for family in ["a", "b"] {
            for policy in ["baseline", "strict", "relaxed"] {
                runs.push(run(
                    family,
                    "field_removed",
                    OracleCompatibilityMode::Backward,
                    policy,
                    "breaking",
                ));
            }
        }
        let report = report(runs);
        let manifest = report
            .promote(&[(
                "field_removed".to_owned(),
                OracleCompatibilityMode::Backward,
                "baseline".to_owned(),
                "reviewed".to_owned(),
            )])
            .unwrap();
        assert!(
            manifest
                .matching(
                    "field_removed",
                    OracleCompatibilityMode::Backward,
                    "baseline",
                    "breaking",
                    "jar",
                    "packs"
                )
                .is_some()
        );
        assert!(
            manifest
                .matching(
                    "field_removed",
                    OracleCompatibilityMode::Forward,
                    "baseline",
                    "breaking",
                    "jar",
                    "packs"
                )
                .is_none()
        );
        assert!(
            manifest
                .matching(
                    "field_removed",
                    OracleCompatibilityMode::Backward,
                    "baseline",
                    "breaking",
                    "changed-jar",
                    "packs"
                )
                .is_none()
        );
    }
}
