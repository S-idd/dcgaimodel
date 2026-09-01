use dcgaimodel::dataset::DatasetSplitConfig;
use dcgaimodel::features::{CompatibilityStatus, realistic_fixtures};
use dcgaimodel::inference::InferenceRuntime;
use dcgaimodel::models::{
    ContractExample, ContractLabels, DcgModel, ModelArtifact, ModelConfig, TrainingMetadata,
    build_dataset,
};
use dcgaimodel::nn::Sgd;
use dcgaimodel::prediction::PredictionKind;
use dcgaimodel::preprocessing::StandardScaler;
use dcgaimodel::training::TrainingConfig;

fn examples() -> Vec<ContractExample> {
    realistic_fixtures()
        .into_iter()
        .map(|fixture| {
            let is_breaking = fixture.deterministic_breaking;
            let breaking = f64::from(is_breaking);
            let incompatible = f64::from(matches!(
                fixture.change.compatibility_status,
                CompatibilityStatus::Incompatible
            ));
            ContractExample {
                contract: fixture.change,
                labels: ContractLabels {
                    breaking_change: breaking,
                    incompatible,
                    risk_score: if is_breaking { 0.8 } else { 0.15 },
                },
            }
        })
        .collect()
}

#[test]
fn realistic_fixture_trains_saves_loads_and_predicts_without_retraining() {
    let examples = examples();
    let dataset = build_dataset(&examples, PredictionKind::BreakingChange).unwrap();
    let split = dataset
        .train_validation_test_split(DatasetSplitConfig::new(0.7, 0.15, 0.15, 2026).unwrap())
        .unwrap();
    let scaler = StandardScaler::fit(split.train()).unwrap();
    let train = scaler.transform_dataset(split.train()).unwrap();
    let validation = scaler.transform_dataset(split.validation()).unwrap();
    let test = scaler.transform_dataset(split.test()).unwrap();
    let mut model =
        DcgModel::with_default_network(PredictionKind::BreakingChange, ModelConfig::default())
            .unwrap();
    model
        .train(
            &train,
            Sgd::new(0.05).unwrap(),
            TrainingConfig::new(8, 2).unwrap(),
        )
        .unwrap();

    let original = model.predict(&test.features()[0]).unwrap();
    let artifact = ModelArtifact::new(
        "dcg-breaking-model-v1",
        model,
        scaler,
        TrainingMetadata {
            optimizer: "sgd".to_owned(),
            epochs: 8,
            batch_size: 2,
            learning_rate: Some("0.05".to_owned()),
            seed: 2026,
            training_samples: train.len(),
            validation_samples: validation.len(),
            test_samples: test.len(),
            dataset_id: "realistic-dcg-fixtures-v1".to_owned(),
        }
        .validate()
        .unwrap(),
    )
    .unwrap();
    let path = std::env::temp_dir().join(format!("dcgaimodel-e2e-{}.json", std::process::id()));
    artifact.save(&path).unwrap();
    drop(artifact);

    let runtime = InferenceRuntime::load(&path).unwrap();
    let loaded = runtime.predict_contract(&examples[0].contract).unwrap();
    let direct = runtime
        .artifact()
        .model()
        .predict(
            &runtime
                .artifact()
                .scaler()
                .transform_vector(
                    &build_dataset(&[examples[0].clone()], PredictionKind::BreakingChange)
                        .unwrap()
                        .features()[0],
                )
                .unwrap(),
        )
        .unwrap();
    assert_eq!(loaded.prediction, direct);
    assert!(matches!(
        original,
        dcgaimodel::prediction::ModelPrediction::Classification { .. }
    ));
    assert_eq!(loaded.feature_version, "dcg-features-v1");
    std::fs::remove_file(path).unwrap();
}
