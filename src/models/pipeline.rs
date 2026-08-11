use super::{ContractExample, DcgDatasetError, DcgModel, ModelConfig, ModelError, build_dataset};
use crate::dataset::{Dataset, DatasetError};
use crate::evaluation::{
    ClassificationMetrics, EvaluationError, RegressionMetrics, evaluate_binary_classification,
    evaluate_regression,
};
use crate::features::{ContractChange, ContractFeatureExtractor, FeatureError, FeatureExtractor};
use crate::nn::Optimizer;
use crate::prediction::{ModelPrediction, PredictionKind};
use crate::preprocessing::StandardScaler;
use crate::training::{TrainingConfig, TrainingHistory};
use std::error::Error;
use std::fmt;

/// Configuration for the high-level DCG training and evaluation pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct DcgPipelineConfig {
    /// DCG target to train and interpret.
    pub kind: PredictionKind,
    /// Fraction of examples retained for deterministic test evaluation.
    pub test_ratio: f64,
    /// Architecture and threshold settings only; training settings live below.
    pub model: ModelConfig,
    /// Existing generic trainer configuration.
    pub training: TrainingConfig,
}

/// Metrics selected by the model task.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvaluationResult {
    /// Binary breaking-change or compatibility metrics.
    Classification(ClassificationMetrics),
    /// Risk-score regression metrics.
    Regression(RegressionMetrics),
}

/// Completed output of the DCG training pipeline.
#[derive(Debug, Clone)]
pub struct DcgPipelineResult {
    /// Trained neural-network model and DCG output interpretation.
    pub model: DcgModel,
    /// Fitted only on training features; use it for all later model inputs.
    pub scaler: StandardScaler,
    /// History from the existing trainer.
    pub training_history: TrainingHistory,
    /// Test features transformed with `scaler` and their original targets.
    pub normalized_test_dataset: Dataset,
    /// Task-appropriate metrics evaluated on the test split.
    pub evaluation: EvaluationResult,
}

impl DcgPipelineResult {
    /// Extracts, transforms with the training scaler, and predicts one contract.
    pub fn predict_contract(
        &self,
        contract: &ContractChange,
    ) -> Result<ModelPrediction, DcgPipelineError> {
        let features = ContractFeatureExtractor::new().extract(contract)?;
        let normalized = self.scaler.transform_vector(&features)?;
        Ok(self.model.predict(&normalized)?)
    }
}

/// Errors returned by the DCG orchestration layer.
#[derive(Debug)]
pub enum DcgPipelineError {
    /// Dataset preparation failed.
    Dataset(DcgDatasetError),
    /// Existing train/test splitting or preprocessing failed.
    GenericDataset(DatasetError),
    /// Contract extraction failed while predicting after training.
    Feature(FeatureError),
    /// Model configuration, training, or prediction failed.
    Model(ModelError),
    /// Evaluation failed.
    Evaluation(EvaluationError),
    /// The model feature dimension did not match extracted DCG features.
    InputDimensionMismatch { expected: usize, actual: usize },
    /// A generated target did not contain exactly one value.
    TargetDimensionMismatch { expected: usize, actual: usize },
    /// A prediction did not match the selected task interpretation.
    UnexpectedPredictionKind,
}

impl fmt::Display for DcgPipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dataset(error) => write!(f, "DCG dataset error: {error}"),
            Self::GenericDataset(error) => write!(f, "Dataset/preprocessing error: {error}"),
            Self::Feature(error) => write!(f, "Feature extraction error: {error}"),
            Self::Model(error) => write!(f, "Model error: {error}"),
            Self::Evaluation(error) => write!(f, "Evaluation error: {error}"),
            Self::InputDimensionMismatch { expected, actual } => write!(
                f,
                "DCG model input dimension mismatch: expected {expected}, got {actual}"
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
            Self::Dataset(error) => Some(error),
            Self::GenericDataset(error) => Some(error),
            Self::Feature(error) => Some(error),
            Self::Model(error) => Some(error),
            Self::Evaluation(error) => Some(error),
            Self::InputDimensionMismatch { .. }
            | Self::TargetDimensionMismatch { .. }
            | Self::UnexpectedPredictionKind => None,
        }
    }
}

impl From<DcgDatasetError> for DcgPipelineError {
    fn from(error: DcgDatasetError) -> Self {
        Self::Dataset(error)
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

impl DcgPipelineConfig {
    /// Runs feature extraction, splitting, training-only scaling, training, prediction, and evaluation.
    pub fn run<O: Optimizer>(
        &self,
        examples: &[ContractExample],
        optimizer: O,
    ) -> Result<DcgPipelineResult, DcgPipelineError> {
        let dataset = build_dataset(examples, self.kind)?;
        if self.model.input_size != dataset.feature_size() {
            return Err(DcgPipelineError::InputDimensionMismatch {
                expected: self.model.input_size,
                actual: dataset.feature_size(),
            });
        }
        let (training_dataset, test_dataset) = dataset.train_test_split(self.test_ratio)?;
        // This is intentionally fit once on training data, then reused for test and future inputs.
        let scaler = StandardScaler::fit(&training_dataset)?;
        let normalized_training = scaler.transform_dataset(&training_dataset)?;
        let normalized_test_dataset = scaler.transform_dataset(&test_dataset)?;
        let mut model = DcgModel::with_default_network(self.kind, self.model.clone())?;
        let training_history = model.train(&normalized_training, optimizer, self.training)?;
        let predictions = model.predict_batch(normalized_test_dataset.features())?;
        let evaluation =
            evaluate_predictions(self.kind, &predictions, normalized_test_dataset.targets())?;
        Ok(DcgPipelineResult {
            model,
            scaler,
            training_history,
            normalized_test_dataset,
            evaluation,
        })
    }
}

fn evaluate_predictions(
    kind: PredictionKind,
    predictions: &[ModelPrediction],
    targets: &[crate::linalg::Vector],
) -> Result<EvaluationResult, DcgPipelineError> {
    if predictions.len() != targets.len() {
        return Err(DcgPipelineError::Evaluation(
            EvaluationError::LengthMismatch {
                predictions: predictions.len(),
                targets: targets.len(),
            },
        ));
    }
    let mut target_values = Vec::with_capacity(targets.len());
    for target in targets {
        if target.len() != 1 {
            return Err(DcgPipelineError::TargetDimensionMismatch {
                expected: 1,
                actual: target.len(),
            });
        }
        target_values.push(target[0]);
    }
    match kind {
        PredictionKind::BreakingChange | PredictionKind::Compatibility => {
            let labels = predictions
                .iter()
                .map(|prediction| {
                    prediction
                        .label()
                        .ok_or(DcgPipelineError::UnexpectedPredictionKind)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(EvaluationResult::Classification(
                evaluate_binary_classification(
                    &labels.iter().map(|label| *label as f64).collect::<Vec<_>>(),
                    &target_values,
                )?,
            ))
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::synthetic_examples;
    use crate::nn::Sgd;

    fn config(kind: PredictionKind) -> DcgPipelineConfig {
        DcgPipelineConfig {
            kind,
            test_ratio: 0.33,
            model: ModelConfig::default(),
            training: TrainingConfig::new(4, 2).unwrap(),
        }
    }

    #[test]
    fn end_to_end_pipeline_trains_predicts_and_evaluates() {
        let result = config(PredictionKind::BreakingChange)
            .run(&synthetic_examples(), Sgd::new(0.05).unwrap())
            .unwrap();
        assert_eq!(result.training_history.len(), 4);
        assert_eq!(result.normalized_test_dataset.feature_size(), 8);
        assert!(matches!(
            result.evaluation,
            EvaluationResult::Classification(_)
        ));
        assert!(
            result
                .training_history
                .epoch_losses()
                .iter()
                .all(|loss| loss.is_finite())
        );
        assert!(
            result
                .predict_contract(&synthetic_examples()[0].contract)
                .is_ok()
        );
    }

    #[test]
    fn uses_training_scaler_for_test_data() {
        let examples = synthetic_examples();
        let source = build_dataset(&examples, PredictionKind::RiskScore).unwrap();
        let (training, _) = source.train_test_split(0.33).unwrap();
        let expected_scaler = StandardScaler::fit(&training).unwrap();
        let result = config(PredictionKind::RiskScore)
            .run(&examples, Sgd::new(0.05).unwrap())
            .unwrap();
        assert_eq!(result.scaler.means(), expected_scaler.means());
        assert_eq!(
            result.scaler.standard_deviations(),
            expected_scaler.standard_deviations()
        );
        assert!(matches!(result.evaluation, EvaluationResult::Regression(_)));
    }
}
