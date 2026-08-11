//! Reusable DCG model wrappers and dataset preparation.

mod data;
mod model;
mod pipeline;

pub use data::{
    ContractExample, ContractLabels, DcgDatasetError, build_dataset, synthetic_examples,
};
pub use model::{DcgModel, ModelConfig, ModelError};
pub use pipeline::{DcgPipelineConfig, DcgPipelineError, DcgPipelineResult, EvaluationResult};
