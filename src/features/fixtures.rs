//! Self-contained DCG-shaped fixtures for training and integration tests.
//!
//! They intentionally do not read the Java DCG project or model raw JSON
//! Schema. Deterministic fixture labels represent policy outcomes and remain
//! authoritative over any learned prediction.

use super::{CompatibilityStatus, ContractChange, ContractMetadata, SemanticVersion};

/// Policy packs represented by the self-contained fixture set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixturePolicyPack {
    /// Default governance behaviour.
    Baseline,
    /// Treats enum additions as breaking.
    Strict,
    /// Ignores enum additions.
    Relaxed,
}

/// The contract-change scenario represented by a fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureScenario {
    NoChange,
    OptionalFieldAddition,
    FieldRemoval,
    TypeChange,
    RequiredFieldAddition,
    EnumAddition,
    EnumRemoval,
    MultipleBreakingChanges,
}

/// A realistic, portable DCG fixture with deterministic governance outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcgFixture {
    /// Stable scenario identifier, not a filesystem path.
    pub id: &'static str,
    /// Contract family used to make grouping risk visible to callers.
    pub contract_family: &'static str,
    /// Policy pack used to establish the authoritative result.
    pub policy_pack: FixturePolicyPack,
    /// Schema-change category.
    pub scenario: FixtureScenario,
    /// Human-readable source version.
    pub old_version: SemanticVersion,
    /// Human-readable target version.
    pub new_version: SemanticVersion,
    /// Extractable ML-engine input.
    pub change: ContractChange,
    /// Deterministic policy result: `true` means breaking.
    pub deterministic_breaking: bool,
}

#[allow(clippy::too_many_arguments)] // Concise private fixture table keeps scenario data readable.
fn fixture(
    id: &'static str,
    family: &'static str,
    pack: FixturePolicyPack,
    scenario: FixtureScenario,
    fields: usize,
    added: usize,
    removed: usize,
    types: usize,
    compatibility: CompatibilityStatus,
    breaking_count: usize,
    consumers: usize,
    breaking: bool,
) -> DcgFixture {
    DcgFixture {
        id,
        contract_family: family,
        policy_pack: pack,
        scenario,
        old_version: SemanticVersion::new(1, 0, 0),
        new_version: SemanticVersion::new(1, 1, 0),
        change: ContractChange {
            contract_name: id.to_owned(),
            number_of_fields: fields,
            fields_added: added,
            fields_removed: removed,
            data_type_changes: types,
            compatibility_status: compatibility,
            schema_version: SemanticVersion::new(1, 1, 0),
            breaking_changes: breaking_count,
            metadata: ContractMetadata {
                dependent_consumers: consumers,
            },
        },
        deterministic_breaking: breaking,
    }
}

/// Returns representative, deterministic DCG contract-change fixtures.
pub fn realistic_fixtures() -> Vec<DcgFixture> {
    vec![
        fixture(
            "orders.created.no-change",
            "orders.created",
            FixturePolicyPack::Baseline,
            FixtureScenario::NoChange,
            3,
            0,
            0,
            0,
            CompatibilityStatus::Compatible,
            0,
            4,
            false,
        ),
        fixture(
            "payments.completed.optional-region",
            "payments.completed",
            FixturePolicyPack::Baseline,
            FixtureScenario::OptionalFieldAddition,
            4,
            1,
            0,
            0,
            CompatibilityStatus::Compatible,
            0,
            9,
            false,
        ),
        fixture(
            "payments.completed.currency-removed",
            "payments.completed",
            FixturePolicyPack::Baseline,
            FixtureScenario::FieldRemoval,
            2,
            0,
            1,
            0,
            CompatibilityStatus::Incompatible,
            1,
            9,
            true,
        ),
        fixture(
            "finance.settlement.amount-type",
            "finance.settlement",
            FixturePolicyPack::Baseline,
            FixtureScenario::TypeChange,
            3,
            0,
            0,
            1,
            CompatibilityStatus::Incompatible,
            1,
            6,
            true,
        ),
        fixture(
            "user.events.required-region",
            "user.events",
            FixturePolicyPack::Baseline,
            FixtureScenario::RequiredFieldAddition,
            3,
            1,
            0,
            0,
            CompatibilityStatus::Incompatible,
            1,
            3,
            true,
        ),
        fixture(
            "orders.created.enum-added-baseline",
            "orders.created",
            FixturePolicyPack::Baseline,
            FixtureScenario::EnumAddition,
            4,
            0,
            0,
            0,
            CompatibilityStatus::Unknown,
            0,
            4,
            false,
        ),
        fixture(
            "orders.created.enum-added-strict",
            "orders.created",
            FixturePolicyPack::Strict,
            FixtureScenario::EnumAddition,
            4,
            0,
            0,
            0,
            CompatibilityStatus::Incompatible,
            1,
            4,
            true,
        ),
        fixture(
            "analytics.events.enum-added-relaxed",
            "analytics.events",
            FixturePolicyPack::Relaxed,
            FixtureScenario::EnumAddition,
            4,
            0,
            0,
            0,
            CompatibilityStatus::Compatible,
            0,
            2,
            false,
        ),
        fixture(
            "analytics.events.enum-removed",
            "analytics.events",
            FixturePolicyPack::Baseline,
            FixtureScenario::EnumRemoval,
            4,
            0,
            0,
            0,
            CompatibilityStatus::Incompatible,
            1,
            2,
            true,
        ),
        fixture(
            "orders.created.multiple-breaks",
            "orders.created",
            FixturePolicyPack::Baseline,
            FixtureScenario::MultipleBreakingChanges,
            2,
            1,
            1,
            1,
            CompatibilityStatus::Incompatible,
            3,
            4,
            true,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_all_required_contract_change_scenarios() {
        let fixtures = realistic_fixtures();
        assert!(
            fixtures
                .iter()
                .any(|fixture| fixture.scenario == FixtureScenario::NoChange)
        );
        assert!(
            fixtures
                .iter()
                .any(|fixture| fixture.scenario == FixtureScenario::OptionalFieldAddition)
        );
        assert!(
            fixtures
                .iter()
                .any(|fixture| fixture.scenario == FixtureScenario::FieldRemoval)
        );
        assert!(
            fixtures
                .iter()
                .any(|fixture| fixture.scenario == FixtureScenario::TypeChange)
        );
        assert!(
            fixtures
                .iter()
                .any(|fixture| fixture.scenario == FixtureScenario::RequiredFieldAddition)
        );
        assert!(
            fixtures
                .iter()
                .any(|fixture| fixture.scenario == FixtureScenario::EnumAddition)
        );
        assert!(
            fixtures
                .iter()
                .any(|fixture| fixture.scenario == FixtureScenario::EnumRemoval)
        );
        assert!(
            fixtures
                .iter()
                .any(|fixture| fixture.scenario == FixtureScenario::MultipleBreakingChanges)
        );
    }
}
