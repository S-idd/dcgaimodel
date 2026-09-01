use super::{
    DcgModel, ModelConfig, ModelError, PreparedDatasetError, PreparedDcgDataset,
    PreparedDcgDatasetSplit, TargetMode, ThreeWayCompatibilityModel,
};
use crate::dataset::{Dataset, DatasetError, DatasetSplitConfig};
use crate::evaluation::{
    BaselineComparisonReport, ClassificationMetrics, EvaluationError, RegressionMetrics,
    ThresholdEvaluation, compare_classification_baselines, evaluate_binary_classification,
    evaluate_regression, evaluate_thresholds,
};
use crate::features::{
    ContractChange, ContractFeatureExtractor, DCG_FEATURE_VERSION, FeatureError, FeatureExtractor,
};
use crate::nn::Optimizer;
use crate::prediction::{ModelPrediction, PredictionKind};
use crate::preprocessing::StandardScaler;
use crate::training::{TrainingConfig, TrainingHistory};
use std::error::Error;
use std::fmt;

/// Configuration for the complete isolated DCG training/evaluation workflow.
#[derive(Debug, Clone, PartialEq)]
pub struct DcgPipelineConfig {
    /// DCG target to train and interpret.
    pub kind: PredictionKind,
    /// Seeded, isolated train/validation/test split configuration.
    pub split: DatasetSplitConfig,
    /// Architecture and fixed deployed threshold.
    pub model: ModelConfig,
    /// Existing generic trainer configuration.
    pub training: TrainingConfig,
    /// Thresholds evaluated on validation data only; none is selected automatically.
    pub validation_thresholds: Vec<f64>,
}

impl DcgPipelineConfig {
    /// Creates a pipeline configuration and validates candidate thresholds.
    pub fn new(
        kind: PredictionKind,
        split: DatasetSplitConfig,
        model: ModelConfig,
        training: TrainingConfig,
        validation_thresholds: Vec<f64>,
    ) -> Result<Self, DcgPipelineError> {
        DatasetSplitConfig::new(
            split.train_ratio,
            split.validation_ratio,
            split.test_ratio,
            split.seed,
        )?;
        for threshold in &validation_thresholds {
            if !threshold.is_finite() || !(0.0..=1.0).contains(threshold) {
                return Err(DcgPipelineError::Evaluation(
                    EvaluationError::InvalidThreshold { value: *threshold },
                ));
            }
        }
        Ok(Self {
            kind,
            split,
            model,
            training,
            validation_thresholds,
        })
    }

    /// Runs grouped splitting, training-only preprocessing, training with
    /// validation-loss tracking, validation threshold evaluation, and isolated
    /// final test evaluation.
    pub fn run<O: Optimizer>(
        &self,
        dataset: &PreparedDcgDataset,
        optimizer: O,
    ) -> Result<DcgPipelineResult, DcgPipelineError> {
        let split = dataset.split_by_group(self.split)?;
        let training_dataset = split.train().to_dataset(self.kind)?;
        let validation_dataset = split.validation().to_dataset(self.kind)?;
        let test_dataset = split.test().to_dataset(self.kind)?;
        if self.model.input_size != training_dataset.feature_size() {
            return Err(DcgPipelineError::InputDimensionMismatch {
                expected: self.model.input_size,
                actual: training_dataset.feature_size(),
            });
        }

        // Fit exactly once on training data, then reuse this immutable state.
        let scaler = StandardScaler::fit(&training_dataset)?;
        let normalized_training = scaler.transform_dataset(&training_dataset)?;
        let normalized_validation_dataset = scaler.transform_dataset(&validation_dataset)?;
        let normalized_test_dataset = scaler.transform_dataset(&test_dataset)?;
        let mut model = DcgModel::with_default_network(self.kind, self.model.clone())?;
        let training_history = model.train_with_validation(
            &normalized_training,
            Some(&normalized_validation_dataset),
            optimizer,
            self.training,
        )?;
        let validation_predictions =
            model.predict_batch(normalized_validation_dataset.features())?;
        let test_predictions = model.predict_batch(normalized_test_dataset.features())?;
        let validation_evaluation = evaluate_predictions(
            self.kind,
            &validation_predictions,
            normalized_validation_dataset.targets(),
        )?;
        let evaluation = evaluate_predictions(
            self.kind,
            &test_predictions,
            normalized_test_dataset.targets(),
        )?;
        let classification_report = match self.kind {
            PredictionKind::BreakingChange | PredictionKind::Compatibility => {
                Some(build_classification_report(
                    self.kind,
                    &self.validation_thresholds,
                    &validation_predictions,
                    normalized_validation_dataset.targets(),
                    split.train(),
                    split.test(),
                    &test_predictions,
                )?)
            }
            PredictionKind::RiskScore => None,
        };
        Ok(DcgPipelineResult {
            feature_version: dataset.feature_version().to_owned(),
            split,
            model,
            scaler,
            training_history,
            normalized_validation_dataset,
            normalized_test_dataset,
            validation_evaluation,
            evaluation,
            classification_report,
        })
    }
}

impl Default for DcgPipelineConfig {
    fn default() -> Self {
        Self {
            kind: PredictionKind::BreakingChange,
            split: DatasetSplitConfig::default(),
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            validation_thresholds: vec![0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        }
    }
}

/// Metrics selected by the model task.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvaluationResult {
    /// Binary breaking-change or compatibility metrics.
    Classification(ClassificationMetrics),
    /// Risk-score regression metrics.
    Regression(RegressionMetrics),
}

/// Classification-only output that keeps threshold and baseline evaluation visible.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationPipelineReport {
    /// Metrics at the model's configured threshold on validation data.
    pub validation_metrics: ClassificationMetrics,
    /// FP/FN trade-offs across caller-selected validation thresholds.
    pub validation_thresholds: Vec<ThresholdEvaluation>,
    /// Transparent test-set baselines versus the advisory neural model.
    pub test_baselines: BaselineComparisonReport,
}

/// Completed output of the cohesive DCG workflow.
#[derive(Debug, Clone)]
pub struct DcgPipelineResult {
    /// Exact feature contract used by the fitted scaler and network.
    pub feature_version: String,
    /// Family-aware raw split. Only `train` was used to fit the scaler.
    pub split: PreparedDcgDatasetSplit,
    /// Trained neural-network model and DCG output interpretation.
    pub model: DcgModel,
    /// Fitted only on training features; use it for all later model inputs.
    pub scaler: StandardScaler,
    /// Training and per-epoch validation loss history.
    pub training_history: TrainingHistory,
    /// Validation features transformed with the training scaler and original targets.
    pub normalized_validation_dataset: Dataset,
    /// Test features transformed with the training scaler and original targets.
    pub normalized_test_dataset: Dataset,
    /// Task-appropriate validation metrics at the configured model threshold.
    pub validation_evaluation: EvaluationResult,
    /// Task-appropriate final metrics on the isolated test partition.
    pub evaluation: EvaluationResult,
    /// Additional reports for binary-classification tasks.
    pub classification_report: Option<ClassificationPipelineReport>,
}

/// Accuracy and confusion counts for canonical SAFE/WARNING/BREAKING classes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThreeWayEvaluation {
    /// Fraction of samples whose argmax class matched the one-hot target.
    pub accuracy: f64,
    /// Rows are true classes and columns are predicted classes in SAFE/WARNING/BREAKING order.
    pub confusion_matrix: [[usize; 3]; 3],
}

/// Result of the grouped, scaled three-way compatibility workflow.
#[derive(Debug, Clone)]
pub struct ThreeWayPipelineResult {
    /// Family-isolated split used for training and evaluation.
    pub split: PreparedDcgDatasetSplit,
    /// Fitted three-class Softmax model.
    pub model: ThreeWayCompatibilityModel,
    /// Scaler fitted on training features only.
    pub scaler: StandardScaler,
    /// Per-epoch Cross-Entropy training and validation loss.
    pub training_history: TrainingHistory,
    /// Final isolated test evaluation.
    pub evaluation: ThreeWayEvaluation,
}

/// Runs family-aware three-way compatibility training with a three-output
/// Softmax head and Cross-Entropy objective.
pub fn run_three_way_compatibility_pipeline<O: Optimizer>(
    dataset: &PreparedDcgDataset,
    split_config: DatasetSplitConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    optimizer: O,
) -> Result<ThreeWayPipelineResult, DcgPipelineError> {
    let split = dataset.split_by_group(split_config)?;
    let train = split
        .train()
        .to_target_dataset(TargetMode::ThreeWayCompatibility)?;
    let validation = split
        .validation()
        .to_target_dataset(TargetMode::ThreeWayCompatibility)?;
    let test = split
        .test()
        .to_target_dataset(TargetMode::ThreeWayCompatibility)?;
    if model_config.input_size != train.feature_size() {
        return Err(DcgPipelineError::InputDimensionMismatch {
            expected: model_config.input_size,
            actual: train.feature_size(),
        });
    }
    let scaler = StandardScaler::fit(&train)?;
    let train = scaler.transform_dataset(&train)?;
    let validation = scaler.transform_dataset(&validation)?;
    let test = scaler.transform_dataset(&test)?;
    let mut model = ThreeWayCompatibilityModel::with_default_network(model_config)?;
    let training_history =
        model.train_with_validation(&train, Some(&validation), optimizer, training_config)?;
    let predictions = test
        .features()
        .iter()
        .map(|features| model.predict(features))
        .collect::<Result<Vec<_>, _>>()?;
    let evaluation = evaluate_three_way(&predictions, test.targets())?;
    Ok(ThreeWayPipelineResult {
        split,
        model,
        scaler,
        training_history,
        evaluation,
    })
}

impl DcgPipelineResult {
    /// Extracts canonical features, applies the stored training scaler, and
    /// predicts one contract without fitting or modifying state.
    pub fn predict_contract(
        &self,
        contract: &ContractChange,
    ) -> Result<ModelPrediction, DcgPipelineError> {
        if self.feature_version != DCG_FEATURE_VERSION {
            return Err(DcgPipelineError::UnsupportedInferenceFeatureVersion {
                actual: self.feature_version.clone(),
            });
        }
        let features = ContractFeatureExtractor::new().extract(contract)?;
        let normalized = self.scaler.transform_vector(&features)?;
        Ok(self.model.predict(&normalized)?)
    }
}

/// Errors returned by the complete DCG orchestration layer.
#[derive(Debug)]
pub enum DcgPipelineError {
    /// Prepared dataset persistence, validation, or family-aware splitting failed.
    PreparedDataset(PreparedDatasetError),
    /// Generic preprocessing failed.
    GenericDataset(DatasetError),
    /// Contract feature extraction failed during post-training prediction.
    Feature(FeatureError),
    /// Model configuration, training, or prediction failed.
    Model(ModelError),
    /// Evaluation failed.
    Evaluation(EvaluationError),
    /// The model feature dimension did not match canonical extracted features.
    InputDimensionMismatch { expected: usize, actual: usize },
    /// Raw `ContractChange` inference exists only for the legacy feature contract.
    UnsupportedInferenceFeatureVersion { actual: String },
    /// A generated target did not contain exactly one value.
    TargetDimensionMismatch { expected: usize, actual: usize },
    /// A prediction did not match the selected DCG task interpretation.
    UnexpectedPredictionKind,
}

impl fmt::Display for DcgPipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreparedDataset(error) => write!(f, "Prepared DCG dataset error: {error}"),
            Self::GenericDataset(error) => write!(f, "Dataset/preprocessing error: {error}"),
            Self::Feature(error) => write!(f, "Feature extraction error: {error}"),
            Self::Model(error) => write!(f, "Model error: {error}"),
            Self::Evaluation(error) => write!(f, "Evaluation error: {error}"),
            Self::InputDimensionMismatch { expected, actual } => write!(
                f,
                "DCG model input dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::UnsupportedInferenceFeatureVersion { actual } => write!(
                f,
                "Raw contract inference is unavailable for feature version {actual}; supply the matching schema-pair feature adapter"
            ),
            Self::TargetDimensionMismatch { expected, actual } => write!(
                f,
                "DCG target dimension mismatch: expected {expected}, got {actual}"
            ),
            Self::UnexpectedPredictionKind => {
                write!(f, "Prediction did not match the selected DCG task.")
            }
        }
    }
}

impl Error for DcgPipelineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::PreparedDataset(error) => Some(error),
            Self::GenericDataset(error) => Some(error),
            Self::Feature(error) => Some(error),
            Self::Model(error) => Some(error),
            Self::Evaluation(error) => Some(error),
            Self::InputDimensionMismatch { .. }
            | Self::UnsupportedInferenceFeatureVersion { .. }
            | Self::TargetDimensionMismatch { .. }
            | Self::UnexpectedPredictionKind => None,
        }
    }
}

impl From<PreparedDatasetError> for DcgPipelineError {
    fn from(error: PreparedDatasetError) -> Self {
        Self::PreparedDataset(error)
    }
}
impl From<DatasetError> for DcgPipelineError {
    fn from(error: DatasetError) -> Self {
        Self::GenericDataset(error)
    }
}
impl From<FeatureError> for DcgPipelineError {
    fn from(error: FeatureError) -> Self {
        Self::Feature(error)
    }
}
impl From<ModelError> for DcgPipelineError {
    fn from(error: ModelError) -> Self {
        Self::Model(error)
    }
}
impl From<EvaluationError> for DcgPipelineError {
    fn from(error: EvaluationError) -> Self {
        Self::Evaluation(error)
    }
}

fn build_classification_report(
    kind: PredictionKind,
    thresholds: &[f64],
    validation_predictions: &[ModelPrediction],
    validation_targets: &[crate::linalg::Vector],
    training_records: &PreparedDcgDataset,
    test_records: &PreparedDcgDataset,
    test_predictions: &[ModelPrediction],
) -> Result<ClassificationPipelineReport, DcgPipelineError> {
    let validation_scores = classification_scores(validation_predictions)?;
    let validation_targets = target_values(validation_targets)?;
    let validation_labels = classification_labels(validation_predictions)?;
    let validation_metrics =
        evaluate_binary_classification(&validation_labels, &validation_targets)?;
    let training_targets = target_values(training_records.to_dataset(kind)?.targets())?;
    let test_dataset = test_records.to_dataset(kind)?;
    let test_targets = target_values(test_dataset.targets())?;
    let test_labels = classification_labels(test_predictions)?;
    Ok(ClassificationPipelineReport {
        validation_metrics,
        validation_thresholds: evaluate_thresholds(
            &validation_scores,
            &validation_targets,
            thresholds,
        )?,
        test_baselines: compare_classification_baselines(
            &training_targets,
            test_dataset.features(),
            &test_targets,
            &test_labels,
            Some(&test_targets),
        )?,
    })
}

fn evaluate_predictions(
    kind: PredictionKind,
    predictions: &[ModelPrediction],
    targets: &[crate::linalg::Vector],
) -> Result<EvaluationResult, DcgPipelineError> {
    let target_values = target_values(targets)?;
    match kind {
        PredictionKind::BreakingChange | PredictionKind::Compatibility => Ok(
            EvaluationResult::Classification(evaluate_binary_classification(
                &classification_labels(predictions)?,
                &target_values,
            )?),
        ),
        PredictionKind::RiskScore => {
            let scores = predictions
                .iter()
                .map(|prediction| {
                    prediction
                        .risk_score()
                        .ok_or(DcgPipelineError::UnexpectedPredictionKind)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(EvaluationResult::Regression(evaluate_regression(
                &scores,
                &target_values,
            )?))
        }
    }
}

fn target_values(targets: &[crate::linalg::Vector]) -> Result<Vec<f64>, DcgPipelineError> {
    targets
        .iter()
        .map(|target| {
            if target.len() != 1 {
                Err(DcgPipelineError::TargetDimensionMismatch {
                    expected: 1,
                    actual: target.len(),
                })
            } else {
                Ok(target[0])
            }
        })
        .collect()
}

/// Evaluates argmax Softmax predictions against one-hot SAFE/WARNING/BREAKING targets.
pub fn evaluate_three_way(
    predictions: &[super::ThreeWayPrediction],
    targets: &[crate::linalg::Vector],
) -> Result<ThreeWayEvaluation, DcgPipelineError> {
    if predictions.len() != targets.len() || targets.is_empty() {
        return Err(DcgPipelineError::TargetDimensionMismatch {
            expected: predictions.len(),
            actual: targets.len(),
        });
    }
    let mut confusion_matrix = [[0; 3]; 3];
    for (prediction, target) in predictions.iter().zip(targets) {
        if target.len() != 3 {
            return Err(DcgPipelineError::TargetDimensionMismatch {
                expected: 3,
                actual: target.len(),
            });
        }
        let actual = target
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .ok_or(DcgPipelineError::TargetDimensionMismatch {
                expected: 3,
                actual: 0,
            })?;
        confusion_matrix[actual][prediction.label.class_index() as usize] += 1;
    }
    let correct = (0..3)
        .map(|index| confusion_matrix[index][index])
        .sum::<usize>();
    Ok(ThreeWayEvaluation {
        accuracy: correct as f64 / targets.len() as f64,
        confusion_matrix,
    })
}

fn classification_labels(predictions: &[ModelPrediction]) -> Result<Vec<f64>, DcgPipelineError> {
    predictions
        .iter()
        .map(|prediction| {
            prediction
                .label()
                .map(f64::from)
                .ok_or(DcgPipelineError::UnexpectedPredictionKind)
        })
        .collect()
}

fn classification_scores(predictions: &[ModelPrediction]) -> Result<Vec<f64>, DcgPipelineError> {
    predictions
        .iter()
        .map(|prediction| {
            prediction
                .probability()
                .ok_or(DcgPipelineError::UnexpectedPredictionKind)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::realistic_fixtures;
    use crate::models::{CompatibilityLabel, PreparedDcgRecord};
    use crate::nn::Sgd;

    #[test]
    fn complete_pipeline_uses_grouped_split_scaling_and_validation_reporting() {
        let dataset =
            PreparedDcgDataset::from_fixtures("fixture-v1", &realistic_fixtures()).unwrap();
        let config = DcgPipelineConfig::new(
            PredictionKind::BreakingChange,
            DatasetSplitConfig::new(0.7, 0.15, 0.15, 22).unwrap(),
            ModelConfig::default(),
            TrainingConfig::new(4, 2).unwrap(),
            vec![0.3, 0.5, 0.7],
        )
        .unwrap();
        let result = config.run(&dataset, Sgd::new(0.05).unwrap()).unwrap();
        assert_eq!(result.training_history.len(), 4);
        assert_eq!(result.training_history.validation_losses().len(), 4);
        assert_eq!(result.normalized_validation_dataset.feature_size(), 8);
        assert!(matches!(
            result.validation_evaluation,
            EvaluationResult::Classification(_)
        ));
        assert!(matches!(
            result.evaluation,
            EvaluationResult::Classification(_)
        ));
        let report = result.classification_report.unwrap();
        assert_eq!(report.validation_thresholds.len(), 3);
        assert_eq!(
            report.test_baselines.deterministic_policy.unwrap().accuracy,
            1.0
        );
        assert_eq!(
            result.scaler.means(),
            StandardScaler::fit(
                &result
                    .split
                    .train()
                    .to_dataset(PredictionKind::BreakingChange)
                    .unwrap()
            )
            .unwrap()
            .means()
        );
    }

    #[test]
    fn rejects_invalid_validation_thresholds() {
        assert!(matches!(
            DcgPipelineConfig::new(
                PredictionKind::BreakingChange,
                DatasetSplitConfig::default(),
                ModelConfig::default(),
                TrainingConfig::default(),
                vec![f64::INFINITY],
            ),
            Err(DcgPipelineError::Evaluation(
                EvaluationError::InvalidThreshold { .. }
            ))
        ));
    }

    #[test]
    fn three_way_pipeline_uses_one_hot_targets_and_cross_entropy_validation_loss() {
        let fixture = &realistic_fixtures()[0];
        let mut records = Vec::new();
        for family in 0..3 {
            for (class, label) in [
                CompatibilityLabel::Safe,
                CompatibilityLabel::Warning,
                CompatibilityLabel::Breaking,
            ]
            .into_iter()
            .enumerate()
            {
                let mut record = PreparedDcgRecord::from_fixture(fixture).unwrap();
                record.record_id = format!("family-{family}-class-{class}");
                record.family_id = format!("family-{family}");
                record.split_group_id = record.family_id.clone();
                record.compatibility_label = Some(label);
                record.labels.breaking_change = label.binary_breaking();
                record.labels.incompatible = label.binary_breaking();
                record.labels.risk_score = match label {
                    CompatibilityLabel::Safe => 0.0,
                    CompatibilityLabel::Warning => 0.5,
                    CompatibilityLabel::Breaking => 1.0,
                };
                records.push(record);
            }
        }
        let dataset = PreparedDcgDataset::new("three-way-test", records).unwrap();
        let result = run_three_way_compatibility_pipeline(
            &dataset,
            DatasetSplitConfig::new(0.7, 0.15, 0.15, 7).unwrap(),
            ModelConfig::new(8, vec![4], 0.5).unwrap(),
            TrainingConfig::new(3, 3).unwrap(),
            Sgd::new(0.05).unwrap(),
        )
        .unwrap();
        assert_eq!(result.training_history.len(), 3);
        assert_eq!(result.training_history.validation_losses().len(), 3);
        assert!(
            result
                .training_history
                .validation_losses()
                .iter()
                .all(|loss| loss.is_finite() && *loss > 0.0)
        );
        assert_eq!(
            result
                .evaluation
                .confusion_matrix
                .iter()
                .flatten()
                .sum::<usize>(),
            result.split.test().len()
        );
    }
}
