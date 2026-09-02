use crate::models::{
    FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_FILE_NAME, FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_SHA256,
    OracleCompatibilityMode,
};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Immutable configuration for the externally built, pinned DCG oracle JAR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleConfig {
    /// Java executable, normally `java`.
    pub java_program: PathBuf,
    /// Explicit path to the pinned contract CLI JAR.
    pub jar_path: PathBuf,
    /// Explicit path to the version-pinned policy-packs JSON file.
    pub policy_packs_path: PathBuf,
}

/// A generated schema transition to stage for the DCG oracle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedPair {
    /// Stable generated contract identifier.
    pub contract_id: String,
    /// Selected policy pack written to staged `metadata.yaml`.
    pub policy_pack: String,
    /// Raw base JSON Schema document.
    pub base_schema: String,
    /// Raw candidate JSON Schema document.
    pub candidate_schema: String,
}

/// Compatibility result emitted by the executable oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleOutcome {
    /// Oracle exit code `0` without a JAR-reported warning section.
    Safe,
    /// Oracle exit code `0` with one or more JAR-reported warnings.
    Warning,
    /// Oracle exit code `1`: deterministic breaking/fail.
    Breaking,
    /// Invalid schema or detected oracle failure; do not train on it.
    Rejected,
}

impl OracleOutcome {
    /// Portable outcome identifier stored in generated evidence.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Warning => "warning",
            Self::Breaking => "breaking",
            Self::Rejected => "rejected",
        }
    }
}

/// Complete captured execution record for one oracle call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleRun {
    /// Oracle compatibility decision determined only from the exit code.
    pub outcome: OracleOutcome,
    /// Raw documented CLI exit code retained independently of the derived outcome.
    pub exit_code: i32,
    /// Oracle standard output retained for audit/debugging.
    pub stdout: String,
    /// Oracle standard error retained for audit/debugging.
    pub stderr: String,
    /// Staged pair directory for successful or rejected inspection.
    pub staged_contract_dir: PathBuf,
    /// Stage at which a rejected invocation failed; absent for accepted labels.
    pub rejection_stage: Option<String>,
    /// First concrete diagnostic emitted for a rejected invocation.
    pub rejection_reason: Option<String>,
}

/// Captured schema-lint preflight for a staged base/candidate pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OraclePreflightRun {
    pub accepted: bool,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub staged_contract_dir: PathBuf,
    pub rejection_stage: Option<String>,
    pub rejection_reason: Option<String>,
}

/// Errors from staging or executing the independent oracle.
#[derive(Debug)]
pub enum GeneratorError {
    /// Required input metadata was empty or unsafe for the staged format.
    InvalidInput { field: &'static str },
    /// Staging files failed.
    Io { message: String },
    /// Source data could not be decoded into a seed schema.
    Source { message: String },
    /// A generated record could not meet the portable prepared-dataset contract.
    Prepared { message: String },
    /// The Java executable could not be started.
    Spawn { message: String },
    /// The process returned an undocumented exit code.
    UnexpectedExitCode { code: Option<i32>, stderr: String },
}

impl fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { field } => write!(f, "Invalid generator input: {field}"),
            Self::Io { message } => write!(f, "Generator staging IO error: {message}"),
            Self::Source { message } => write!(f, "Generator source error: {message}"),
            Self::Prepared { message } => write!(f, "Generator prepared-data error: {message}"),
            Self::Spawn { message } => write!(f, "Could not start pinned oracle: {message}"),
            Self::UnexpectedExitCode { code, stderr } => write!(
                f,
                "Pinned oracle returned unexpected exit code {:?}: {}",
                code, stderr
            ),
        }
    }
}

impl Error for GeneratorError {}

/// Invokes the executable oracle without linking to or reimplementing Java logic.
#[derive(Debug, Clone)]
pub struct PinnedOracle {
    config: OracleConfig,
    jar_sha256: String,
    policy_packs_sha256: String,
}

impl PinnedOracle {
    /// Creates a production adapter after verifying that the pinned input
    /// files exist. The audit-only FIELD_REMOVED fixture is rejected by hash,
    /// even if it is renamed or copied elsewhere.
    pub fn new(config: OracleConfig) -> Result<Self, GeneratorError> {
        Self::new_with_policy_scope(config, false)
    }

    /// Creates the narrowly scoped adapter used only by the removal-severity
    /// mechanism audit. It requires both the fixture's exact name and hash.
    pub fn new_policy_mechanism_audit(config: OracleConfig) -> Result<Self, GeneratorError> {
        Self::new_with_policy_scope(config, true)
    }

    fn new_with_policy_scope(
        config: OracleConfig,
        allow_audit_fixture: bool,
    ) -> Result<Self, GeneratorError> {
        for (field, path) in [
            ("jar_path", &config.jar_path),
            ("policy_packs_path", &config.policy_packs_path),
        ] {
            if !path.is_file() {
                return Err(GeneratorError::InvalidInput { field });
            }
        }
        let jar_contents = fs::read(&config.jar_path).map_err(|error| GeneratorError::Io {
            message: error.to_string(),
        })?;
        let policy_contents =
            fs::read(&config.policy_packs_path).map_err(|error| GeneratorError::Io {
                message: error.to_string(),
            })?;
        let policy_packs_sha256 = format!("{:x}", Sha256::digest(policy_contents));
        validate_policy_scope(
            &config.policy_packs_path,
            &policy_packs_sha256,
            allow_audit_fixture,
        )?;
        Ok(Self {
            config,
            jar_sha256: format!("{:x}", Sha256::digest(jar_contents)),
            policy_packs_sha256,
        })
    }

    /// SHA-256 identity of the exact executable used to label records.
    pub fn jar_sha256(&self) -> &str {
        &self.jar_sha256
    }

    /// SHA-256 identity of the exact approved policy-pack file used by calls.
    pub fn policy_packs_sha256(&self) -> &str {
        &self.policy_packs_sha256
    }

    /// Returns the immutable policy-pack path used to stage every oracle call.
    /// Feature extraction may read this same pinned file for non-label policy
    /// configuration context; it never reads an oracle outcome.
    pub fn policy_packs_path(&self) -> &Path {
        &self.config.policy_packs_path
    }

    /// Stages the supplied pair beneath `workspace/contracts/generated-contract`
    /// and runs `check-compat --mode BACKWARD` against the JAR.
    ///
    /// The caller owns `workspace`; staging uses a per-run subdirectory and
    /// never writes into the Java repository or source data corpus.
    pub fn check_backward(
        &self,
        workspace: &Path,
        pair: &GeneratedPair,
    ) -> Result<OracleRun, GeneratorError> {
        self.check(workspace, pair, OracleCompatibilityMode::Backward)
    }

    /// Stages a pair and delegates the requested compatibility direction to
    /// the executable JAR. Rust never reimplements the mode semantics.
    pub fn check(
        &self,
        workspace: &Path,
        pair: &GeneratedPair,
        mode: OracleCompatibilityMode,
    ) -> Result<OracleRun, GeneratorError> {
        validate_pair(pair)?;
        let staged_contract_dir = self.stage_pair(workspace, pair, mode)?;
        let output = Command::new(&self.config.java_program)
            .arg("-jar")
            .arg(&self.config.jar_path)
            .arg("check-compat")
            .arg("--base")
            .arg(staged_contract_dir.join("v1.json"))
            .arg("--candidate")
            .arg(staged_contract_dir.join("v2.json"))
            .arg("--mode")
            .arg(mode.as_str())
            .output()
            .map_err(|error| GeneratorError::Spawn {
                message: error.to_string(),
            })?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let exit_code = output.status.code();
        let (outcome, rejection_stage) = match exit_code {
            Some(0) => (outcome_from_success_stdout(&stdout), None),
            // The documented exit-1 path is a compatibility failure. The
            // pinned JAR can also leak an uncaught JVM exception through that
            // code. An empty decision stream plus an exception banner is not
            // a compatibility label, so reject it rather than training a
            // model to reproduce an oracle crash.
            Some(1) if has_fatal_oracle_runtime_failure(&stdout, &stderr) => {
                (OracleOutcome::Rejected, Some("jar_runtime".to_owned()))
            }
            Some(1) => (OracleOutcome::Breaking, None),
            Some(2) => (
                OracleOutcome::Rejected,
                Some("compatibility_validation".to_owned()),
            ),
            code => {
                return Err(GeneratorError::UnexpectedExitCode { code, stderr });
            }
        };
        let rejection_reason = rejection_stage
            .as_ref()
            .map(|_| first_diagnostic(&stdout, &stderr));
        Ok(OracleRun {
            outcome,
            exit_code: exit_code.expect("documented oracle exit code was matched"),
            stdout,
            stderr,
            staged_contract_dir,
            rejection_reason,
            rejection_stage,
        })
    }

    /// Lints both version files before a mutation enters the policy/mode
    /// matrix. This is deliberately separate from compatibility evaluation.
    pub fn preflight(
        &self,
        workspace: &Path,
        pair: &GeneratedPair,
    ) -> Result<OraclePreflightRun, GeneratorError> {
        validate_pair(pair)?;
        let staged_contract_dir =
            self.stage_pair(workspace, pair, OracleCompatibilityMode::Forward)?;
        let output = Command::new(&self.config.java_program)
            .arg("-jar")
            .arg(&self.config.jar_path)
            .arg("lint")
            .arg("--path")
            .arg(&staged_contract_dir)
            .output()
            .map_err(|error| GeneratorError::Spawn {
                message: error.to_string(),
            })?;
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let exit_code = output.status.code();
        let accepted = match exit_code {
            Some(0) => true,
            Some(1) => false,
            code => {
                return Err(GeneratorError::UnexpectedExitCode { code, stderr });
            }
        };
        let rejection_reason = (!accepted).then(|| first_diagnostic(&stdout, &stderr));
        Ok(OraclePreflightRun {
            accepted,
            exit_code: exit_code.expect("documented lint exit code was matched"),
            rejection_stage: (!accepted).then(|| "schema_preflight".to_owned()),
            rejection_reason,
            stdout,
            stderr,
            staged_contract_dir,
        })
    }

    fn stage_pair(
        &self,
        workspace: &Path,
        pair: &GeneratedPair,
        mode: OracleCompatibilityMode,
    ) -> Result<PathBuf, GeneratorError> {
        let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = workspace.join(format!("oracle-run-{}-{sequence}", std::process::id()));
        let contracts = root.join("contracts");
        // The Java lint command enforces the same lowercase dot-separated
        // contract-id convention as production contract directories.
        let contract_dir = contracts.join("generated.contract");
        fs::create_dir_all(&contract_dir).map_err(|error| GeneratorError::Io {
            message: error.to_string(),
        })?;
        fs::copy(
            &self.config.policy_packs_path,
            contracts.join("policy-packs.json"),
        )
        .map_err(|error| GeneratorError::Io {
            message: error.to_string(),
        })?;
        fs::write(
            contract_dir.join("metadata.yaml"),
            format!(
                "ownerTeam: generated\ndomain: generated\ncompatibilityMode: {}\npolicyPack: {}\n",
                mode.as_str(),
                pair.policy_pack
            ),
        )
        .map_err(|error| GeneratorError::Io {
            message: error.to_string(),
        })?;
        fs::write(contract_dir.join("v1.json"), &pair.base_schema).map_err(|error| {
            GeneratorError::Io {
                message: error.to_string(),
            }
        })?;
        fs::write(contract_dir.join("v2.json"), &pair.candidate_schema).map_err(|error| {
            GeneratorError::Io {
                message: error.to_string(),
            }
        })?;
        Ok(contract_dir)
    }
}

fn validate_policy_scope(
    path: &Path,
    sha256: &str,
    allow_audit_fixture: bool,
) -> Result<(), GeneratorError> {
    let is_audit_fixture = sha256 == FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_SHA256;
    if !allow_audit_fixture && is_audit_fixture {
        return Err(GeneratorError::InvalidInput {
            field: "audit_only_policy_fixture_forbidden_in_production",
        });
    }
    if allow_audit_fixture {
        let has_exact_name = path.file_name().and_then(|name| name.to_str())
            == Some(FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_FILE_NAME);
        if !is_audit_fixture || !has_exact_name {
            return Err(GeneratorError::InvalidInput {
                field: "policy_mechanism_audit_requires_exact_fixture_name_and_hash",
            });
        }
    }
    Ok(())
}

fn first_diagnostic(stdout: &str, stderr: &str) -> String {
    stderr
        .lines()
        .chain(stdout.lines())
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("oracle rejected without a diagnostic")
        .to_owned()
}

/// The pinned CLI has only three exit codes. Within an accepted exit-0 result,
/// its own `Warnings:` section distinguishes a clean safe result from a
/// policy warning; Rust never infers warnings from the proposed mutation.
fn outcome_from_success_stdout(stdout: &str) -> OracleOutcome {
    if stdout.lines().any(|line| {
        line.trim_start().starts_with("Warnings: [") && !line.trim_end().ends_with("[]")
    }) {
        OracleOutcome::Warning
    } else {
        OracleOutcome::Safe
    }
}

/// Identifies an uncaught JVM failure that cannot be interpreted as the
/// documented breaking-change decision. This deliberately requires both an
/// empty decision stream and the JVM's exception banner, so normal exit-1
/// `Schema compatibility: FAIL` output remains a breaking label.
pub fn has_fatal_oracle_runtime_failure(stdout: &str, stderr: &str) -> bool {
    stdout.trim().is_empty()
        && stderr
            .lines()
            .any(|line| line.trim_start().starts_with("Exception in thread"))
}

fn validate_pair(pair: &GeneratedPair) -> Result<(), GeneratorError> {
    if pair.contract_id.trim().is_empty() {
        return Err(GeneratorError::InvalidInput {
            field: "contract_id",
        });
    }
    if pair.policy_pack.trim().is_empty()
        || !pair.policy_pack.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == '-'
        })
    {
        return Err(GeneratorError::InvalidInput {
            field: "policy_pack",
        });
    }
    if pair.base_schema.trim().is_empty() {
        return Err(GeneratorError::InvalidInput {
            field: "base_schema",
        });
    }
    if pair.candidate_schema.trim().is_empty() {
        return Err(GeneratorError::InvalidInput {
            field: "candidate_schema",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{MutationKind, candidates_for_schema};
    use std::collections::BTreeSet;

    #[test]
    fn rejects_unsafe_staging_metadata_before_running_an_oracle() {
        let pair = GeneratedPair {
            contract_id: "generated".to_owned(),
            policy_pack: "baseline\nother: value".to_owned(),
            base_schema: "{}".to_owned(),
            candidate_schema: "{}".to_owned(),
        };
        assert!(matches!(
            validate_pair(&pair),
            Err(GeneratorError::InvalidInput {
                field: "policy_pack"
            })
        ));
    }

    #[test]
    fn production_rejects_audit_fixture_identity_even_if_renamed() {
        let renamed = Path::new("renamed-policy-packs.json");
        assert!(matches!(
            validate_policy_scope(renamed, FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_SHA256, false),
            Err(GeneratorError::InvalidInput {
                field: "audit_only_policy_fixture_forbidden_in_production"
            })
        ));
    }

    #[test]
    fn audit_constructor_requires_exact_fixture_name_and_hash() {
        let exact = Path::new(FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_FILE_NAME);
        assert!(
            validate_policy_scope(exact, FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_SHA256, true).is_ok()
        );
        assert!(validate_policy_scope(exact, &"0".repeat(64), true).is_err());
        assert!(
            validate_policy_scope(
                Path::new("renamed-audit-fixture.json"),
                FIELD_REMOVED_SEVERITY_AUDIT_FIXTURE_SHA256,
                true
            )
            .is_err()
        );
    }

    #[test]
    fn distinguishes_safe_and_warning_exit_zero_output_from_the_jar() {
        assert_eq!(
            outcome_from_success_stdout("Schema compatibility: PASS\n"),
            OracleOutcome::Safe
        );
        assert_eq!(
            outcome_from_success_stdout(
                "Schema compatibility: PASS\nWarnings: [Enum value added: status.NEW]\n"
            ),
            OracleOutcome::Warning
        );
    }

    #[test]
    fn rejects_an_exit_one_jvm_crash_without_rejecting_a_real_fail_decision() {
        assert!(has_fatal_oracle_runtime_failure(
            "",
            "Exception in thread \"main\" java.lang.StackOverflowError\n"
        ));
        assert!(!has_fatal_oracle_runtime_failure(
            "Schema compatibility: FAIL\nBreaking changes: [Field removed: id]\n",
            ""
        ));
    }

    /// Executes every rich hard-constraint candidate against a deliberately
    /// supplied pinned JAR. It is opt-in because the JAR and policy-pack file
    /// are external build artefacts, not Cargo dependencies. Run with:
    ///
    /// `DCG_PINNED_ORACLE_JAR=... DCG_POLICY_PACKS=... cargo test
    /// generation::oracle::tests::pinned_jar_labels_every_hard_constraint_variant
    /// -- --ignored --nocapture`
    #[test]
    #[ignore = "requires the explicit pinned JAR and policy-pack paths"]
    fn pinned_jar_labels_every_hard_constraint_variant() {
        let jar_path = std::env::var_os("DCG_PINNED_ORACLE_JAR")
            .map(PathBuf::from)
            .expect("set DCG_PINNED_ORACLE_JAR to the executable contract-cli JAR");
        let policy_packs_path = std::env::var_os("DCG_POLICY_PACKS")
            .map(PathBuf::from)
            .expect("set DCG_POLICY_PACKS to the approved policy-packs JSON");
        let oracle = PinnedOracle::new(OracleConfig {
            java_program: PathBuf::from("java"),
            jar_path,
            policy_packs_path,
        })
        .expect("the requested JAR and policy packs must be readable");
        let schema = r#"{
          "type": "object",
          "properties": {
            "amount": {"type": "number"},
            "name": {"type": "string"},
            "payload": {"type": "object", "properties": {"kind": {"type": "string"}}},
            "tags": {"type": "array", "items": {"type": "string"}}
          }
        }"#;
        let expected_variants = BTreeSet::from([
            "tighten-numeric-minimum-0",
            "tighten-numeric-maximum-0",
            "tighten-numeric-multiple-of-0",
            "tighten-numeric-minimum-multiple-of-0",
            "tighten-string-min-length-1",
            "tighten-string-max-length-1",
            "tighten-string-pattern-1",
            "tighten-string-min-length-pattern-1",
            "tighten-object-min-properties-2",
            "tighten-object-max-properties-2",
            "tighten-object-min-max-properties-2",
            "tighten-array-min-items-3",
            "tighten-array-max-items-3",
            "tighten-array-unique-items-3",
            "tighten-array-min-items-unique-items-3",
        ]);
        let candidates = candidates_for_schema(schema)
            .expect("the hard-constraint fixture must remain valid JSON")
            .into_iter()
            .filter(|candidate| candidate.kind == MutationKind::ConstraintTightened)
            .collect::<Vec<_>>();
        let actual_variants = candidates
            .iter()
            .map(|candidate| candidate.variant.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(actual_variants, expected_variants);

        let workspace = PathBuf::from("target/pinned-oracle-hard-constraint-test");
        for candidate in candidates {
            let run = oracle
                .check_backward(
                    &workspace,
                    &GeneratedPair {
                        contract_id: format!("hard-constraint-{}", candidate.variant),
                        policy_pack: "baseline".to_owned(),
                        base_schema: schema.to_owned(),
                        candidate_schema: candidate.candidate_schema,
                    },
                )
                .unwrap_or_else(|error| panic!("{}: {error}", candidate.variant));
            assert_eq!(
                run.outcome,
                OracleOutcome::Breaking,
                "{} exit={} stdout={} stderr={}",
                candidate.variant,
                run.exit_code,
                run.stdout,
                run.stderr
            );
        }
    }

    /// Verifies that the approved V5 composition matrix is interpreted by the
    /// executable JAR as the declared independent enum and constraint actions.
    #[test]
    #[ignore = "requires the explicit pinned JAR and V5 compositional policy-pack paths"]
    fn pinned_jar_resolves_every_compositional_policy_profile() {
        let jar_path = std::env::var_os("DCG_PINNED_ORACLE_JAR")
            .map(PathBuf::from)
            .expect("set DCG_PINNED_ORACLE_JAR to the executable contract-cli JAR");
        let policy_packs_path = std::env::var_os("DCG_POLICY_PACKS")
            .map(PathBuf::from)
            .expect("set DCG_POLICY_PACKS to the approved policy-packs JSON");
        let oracle = PinnedOracle::new(OracleConfig {
            java_program: PathBuf::from("java"),
            jar_path,
            policy_packs_path,
        })
        .expect("the requested JAR and policy packs must be readable");
        let schema = r#"{
          "type": "object",
          "properties": {
            "amount": {"type": "number"},
            "status": {"type": "string", "enum": ["old"]}
          }
        }"#;
        let candidates =
            candidates_for_schema(schema).expect("the composition fixture must remain valid JSON");
        let enum_candidate = candidates
            .iter()
            .find(|candidate| candidate.kind == MutationKind::EnumValueAdded)
            .expect("fixture must propose an enum-value addition");
        let constraint_candidate = candidates
            .iter()
            .find(|candidate| candidate.variant == "tighten-numeric-minimum-0")
            .expect("fixture must propose numeric constraint tightening");
        let profiles = [
            (
                "composition-enum-ignore-constraint-ignore",
                OracleOutcome::Safe,
                OracleOutcome::Safe,
            ),
            (
                "composition-enum-ignore-constraint-warning",
                OracleOutcome::Safe,
                OracleOutcome::Warning,
            ),
            (
                "composition-enum-ignore-constraint-breaking",
                OracleOutcome::Safe,
                OracleOutcome::Breaking,
            ),
            (
                "composition-enum-warning-constraint-ignore",
                OracleOutcome::Warning,
                OracleOutcome::Safe,
            ),
            (
                "composition-enum-warning-constraint-warning",
                OracleOutcome::Warning,
                OracleOutcome::Warning,
            ),
            (
                "composition-enum-warning-constraint-breaking",
                OracleOutcome::Warning,
                OracleOutcome::Breaking,
            ),
            (
                "composition-enum-breaking-constraint-ignore",
                OracleOutcome::Breaking,
                OracleOutcome::Safe,
            ),
            (
                "composition-enum-breaking-constraint-warning",
                OracleOutcome::Breaking,
                OracleOutcome::Warning,
            ),
            (
                "composition-enum-breaking-constraint-breaking",
                OracleOutcome::Breaking,
                OracleOutcome::Breaking,
            ),
        ];
        let workspace = PathBuf::from("target/pinned-oracle-compositional-policy-test");
        for (policy_pack, expected_enum, expected_constraint) in profiles {
            for (candidate, expected) in [
                (enum_candidate, expected_enum),
                (constraint_candidate, expected_constraint),
            ] {
                let run = oracle
                    .check_backward(
                        &workspace,
                        &GeneratedPair {
                            contract_id: format!("composition-{policy_pack}-{}", candidate.variant),
                            policy_pack: policy_pack.to_owned(),
                            base_schema: schema.to_owned(),
                            candidate_schema: candidate.candidate_schema.clone(),
                        },
                    )
                    .unwrap_or_else(|error| panic!("{policy_pack}/{}: {error}", candidate.variant));
                assert_eq!(
                    run.outcome, expected,
                    "{policy_pack}/{} exit={} stdout={} stderr={}",
                    candidate.variant, run.exit_code, run.stdout, run.stderr
                );
            }
        }
    }
}
