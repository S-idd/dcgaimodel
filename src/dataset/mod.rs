use crate::linalg::Vector;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

/// Errors that can occur while loading or preparing datasets.
#[derive(Debug, Clone, PartialEq)]
pub enum DatasetError {
    /// The dataset contains no rows.
    EmptyDataset,

    /// The CSV input contains no data rows.
    EmptyCsv,

    /// The number of target columns is invalid for a row.
    InvalidTargetColumnCount {
        target_columns: usize,
        total_columns: usize,
    },

    /// A CSV row had a different number of columns than previous rows.
    InconsistentColumnCount {
        row: usize,
        expected: usize,
        actual: usize,
    },

    /// A CSV value could not be parsed as a floating-point number.
    ParseFloat {
        row: usize,
        column: usize,
        value: String,
    },

    /// Feature and target row counts do not match.
    RowCountMismatch { features: usize, targets: usize },

    /// A feature vector has an unexpected dimension.
    FeatureDimensionMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },

    /// Tried to read a row outside the dataset bounds.
    IndexOutOfBounds { index: usize, length: usize },

    /// Batch size must be at least 1.
    InvalidBatchSize { batch_size: usize },

    /// Test ratio must be finite and in the open range `0..1`.
    InvalidSplitRatio { ratio: f64 },

    /// Train, validation, and test ratios were not a usable complete split.
    InvalidSplitConfiguration {
        train_ratio: f64,
        validation_ratio: f64,
        test_ratio: f64,
    },

    /// A dataset value was NaN or infinite.
    NonFiniteValue {
        row: usize,
        column: usize,
        value: f64,
    },

    /// File IO failed.
    IoError { message: String },
}

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DatasetError::EmptyDataset => write!(f, "Dataset must contain at least one row."),
            DatasetError::EmptyCsv => write!(f, "CSV input contains no data rows."),
            DatasetError::InvalidTargetColumnCount {
                target_columns,
                total_columns,
            } => write!(
                f,
                "Invalid target column count: target_columns = {}, total_columns = {}",
                target_columns, total_columns
            ),
            DatasetError::InconsistentColumnCount {
                row,
                expected,
                actual,
            } => write!(
                f,
                "CSV row {} has {} columns, expected {}",
                row, actual, expected
            ),
            DatasetError::ParseFloat { row, column, value } => write!(
                f,
                "Could not parse CSV value '{}' at row {}, column {}",
                value, row, column
            ),
            DatasetError::RowCountMismatch { features, targets } => write!(
                f,
                "Feature and target row counts differ: features = {}, targets = {}",
                features, targets
            ),
            DatasetError::FeatureDimensionMismatch {
                row,
                expected,
                actual,
            } => write!(
                f,
                "Feature row {} has dimension {}, expected {}",
                row, actual, expected
            ),
            DatasetError::IndexOutOfBounds { index, length } => write!(
                f,
                "Index {} is out of bounds for dataset of length {}",
                index, length
            ),
            DatasetError::InvalidBatchSize { batch_size } => {
                write!(f, "Batch size must be at least 1: {}", batch_size)
            }
            DatasetError::InvalidSplitRatio { ratio } => {
                write!(
                    f,
                    "Split ratio must be finite and in the range 0..1: {}",
                    ratio
                )
            }
            DatasetError::InvalidSplitConfiguration {
                train_ratio,
                validation_ratio,
                test_ratio,
            } => write!(
                f,
                "Invalid train/validation/test ratios: {train_ratio}/{validation_ratio}/{test_ratio}"
            ),
            DatasetError::NonFiniteValue { row, column, value } => write!(
                f,
                "Dataset value at row {row}, column {column} is not finite: {value}"
            ),
            DatasetError::IoError { message } => write!(f, "Dataset IO error: {}", message),
        }
    }
}

impl Error for DatasetError {}

impl From<std::io::Error> for DatasetError {
    fn from(error: std::io::Error) -> Self {
        DatasetError::IoError {
            message: error.to_string(),
        }
    }
}

/// A supervised dataset containing feature vectors and target vectors.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    features: Vec<Vector>,
    targets: Vec<Vector>,
}

/// Reproducible configuration for an isolated train/validation/test split.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DatasetSplitConfig {
    /// Fraction of rows allocated to training.
    pub train_ratio: f64,
    /// Fraction of rows allocated to validation/model selection.
    pub validation_ratio: f64,
    /// Fraction of rows reserved for final testing.
    pub test_ratio: f64,
    /// Seed used by the internal deterministic row permutation.
    pub seed: u64,
}

impl DatasetSplitConfig {
    /// Creates a split configuration with ratios that sum to one.
    pub fn new(
        train_ratio: f64,
        validation_ratio: f64,
        test_ratio: f64,
        seed: u64,
    ) -> Result<Self, DatasetError> {
        let ratios = [train_ratio, validation_ratio, test_ratio];
        if ratios
            .iter()
            .any(|ratio| !ratio.is_finite() || *ratio <= 0.0)
            || (ratios.iter().sum::<f64>() - 1.0).abs() > 1e-12
        {
            return Err(DatasetError::InvalidSplitConfiguration {
                train_ratio,
                validation_ratio,
                test_ratio,
            });
        }
        Ok(Self {
            train_ratio,
            validation_ratio,
            test_ratio,
            seed,
        })
    }
}

impl Default for DatasetSplitConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.70,
            validation_ratio: 0.15,
            test_ratio: 0.15,
            seed: 0,
        }
    }
}

/// Three isolated partitions produced from a seeded deterministic permutation.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetSplit {
    train: Dataset,
    validation: Dataset,
    test: Dataset,
}

impl DatasetSplit {
    /// Returns the training partition used to fit preprocessing and weights.
    pub fn train(&self) -> &Dataset {
        &self.train
    }

    /// Returns the validation partition used for model or threshold selection.
    pub fn validation(&self) -> &Dataset {
        &self.validation
    }

    /// Returns the final isolated test partition.
    pub fn test(&self) -> &Dataset {
        &self.test
    }
}

impl Dataset {
    /// Creates a supervised dataset.
    pub fn new(features: Vec<Vector>, targets: Vec<Vector>) -> Result<Self, DatasetError> {
        validate_dataset(&features, &targets)?;

        Ok(Self { features, targets })
    }

    /// Loads a supervised dataset from CSV text.
    ///
    /// Target columns are taken from the end of each row.
    pub fn from_csv_str(
        csv: &str,
        target_columns: usize,
        has_header: bool,
    ) -> Result<Self, DatasetError> {
        let data_lines = csv
            .lines()
            .enumerate()
            .filter_map(|(index, line)| {
                let trimmed = line.trim();

                if trimmed.is_empty() || (has_header && index == 0) {
                    None
                } else {
                    Some((index + 1, trimmed))
                }
            })
            .collect::<Vec<(usize, &str)>>();

        if data_lines.is_empty() {
            return Err(DatasetError::EmptyCsv);
        }

        let mut expected_columns = None;
        let mut features = Vec::with_capacity(data_lines.len());
        let mut targets = Vec::with_capacity(data_lines.len());

        for (row_number, line) in data_lines {
            let values = parse_csv_row(line, row_number)?;
            let total_columns = values.len();

            if let Some(expected) = expected_columns {
                if total_columns != expected {
                    return Err(DatasetError::InconsistentColumnCount {
                        row: row_number,
                        expected,
                        actual: total_columns,
                    });
                }
            } else {
                expected_columns = Some(total_columns);
            }

            if target_columns == 0 || target_columns >= total_columns {
                return Err(DatasetError::InvalidTargetColumnCount {
                    target_columns,
                    total_columns,
                });
            }

            let feature_columns = total_columns - target_columns;

            features.push(Vector::new(values[..feature_columns].to_vec()));
            targets.push(Vector::new(values[feature_columns..].to_vec()));
        }

        Self::new(features, targets)
    }

    /// Loads a supervised dataset from a CSV file.
    pub fn from_csv_path<P: AsRef<Path>>(
        path: P,
        target_columns: usize,
        has_header: bool,
    ) -> Result<Self, DatasetError> {
        let csv = fs::read_to_string(path)?;

        Self::from_csv_str(&csv, target_columns, has_header)
    }

    /// Returns the number of rows.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Returns true when the dataset has no rows.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Returns all feature vectors.
    pub fn features(&self) -> &[Vector] {
        &self.features
    }

    /// Returns all target vectors.
    pub fn targets(&self) -> &[Vector] {
        &self.targets
    }

    /// Returns the feature and target vectors at an index.
    pub fn get(&self, index: usize) -> Result<(&Vector, &Vector), DatasetError> {
        if index >= self.len() {
            return Err(DatasetError::IndexOutOfBounds {
                index,
                length: self.len(),
            });
        }

        Ok((&self.features[index], &self.targets[index]))
    }

    /// Returns the number of feature values per row.
    pub fn feature_size(&self) -> usize {
        self.features.first().map(Vector::len).unwrap_or(0)
    }

    /// Returns the number of target values per row.
    pub fn target_size(&self) -> usize {
        self.targets.first().map(Vector::len).unwrap_or(0)
    }

    /// Splits this dataset into deterministic mini-batches.
    pub fn batches(&self, batch_size: usize) -> Result<Vec<DatasetBatch>, DatasetError> {
        if batch_size == 0 {
            return Err(DatasetError::InvalidBatchSize { batch_size });
        }

        Ok((0..self.len())
            .step_by(batch_size)
            .map(|start| {
                let end = (start + batch_size).min(self.len());

                DatasetBatch::new(
                    self.features[start..end].to_vec(),
                    self.targets[start..end].to_vec(),
                )
            })
            .collect())
    }

    /// Splits this dataset into train and test sets while preserving row order.
    pub fn train_test_split(&self, test_ratio: f64) -> Result<(Dataset, Dataset), DatasetError> {
        if !test_ratio.is_finite() || !(0.0..1.0).contains(&test_ratio) {
            return Err(DatasetError::InvalidSplitRatio { ratio: test_ratio });
        }

        if self.len() < 2 {
            return Err(DatasetError::EmptyDataset);
        }

        let test_size = ((self.len() as f64) * test_ratio).round() as usize;
        let test_size = test_size.clamp(1, self.len() - 1);
        let train_size = self.len() - test_size;

        let train = Dataset::new(
            self.features[..train_size].to_vec(),
            self.targets[..train_size].to_vec(),
        )?;
        let test = Dataset::new(
            self.features[train_size..].to_vec(),
            self.targets[train_size..].to_vec(),
        )?;

        Ok((train, test))
    }

    /// Creates reproducible train, validation, and test partitions.
    ///
    /// The seed controls a local deterministic permutation, rather than any
    /// global random state. For related contract versions, callers should use
    /// a group-aware split outside this generic row container; fixtures expose
    /// their `contract_family` specifically so that risk is not hidden.
    pub fn train_validation_test_split(
        &self,
        config: DatasetSplitConfig,
    ) -> Result<DatasetSplit, DatasetError> {
        DatasetSplitConfig::new(
            config.train_ratio,
            config.validation_ratio,
            config.test_ratio,
            config.seed,
        )?;
        if self.len() < 3 {
            return Err(DatasetError::EmptyDataset);
        }

        let mut indices = (0..self.len()).collect::<Vec<_>>();
        let mut state = config.seed;
        for index in (1..indices.len()).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let swap_index = (state as usize) % (index + 1);
            indices.swap(index, swap_index);
        }

        let ratios = [
            config.train_ratio,
            config.validation_ratio,
            config.test_ratio,
        ];
        let mut counts = ratios.map(|ratio| (ratio * self.len() as f64).floor() as usize);
        let mut remaining = self.len() - counts.iter().sum::<usize>();
        while remaining > 0 {
            let index = (0..3)
                .max_by(|left, right| {
                    let left_fraction = ratios[*left] * self.len() as f64 - counts[*left] as f64;
                    let right_fraction = ratios[*right] * self.len() as f64 - counts[*right] as f64;
                    left_fraction.total_cmp(&right_fraction)
                })
                .unwrap_or(0);
            counts[index] += 1;
            remaining -= 1;
        }
        for count in &mut counts {
            if *count == 0 {
                *count = 1;
            }
        }
        while counts.iter().sum::<usize>() > self.len() {
            let index = (0..3)
                .filter(|index| counts[*index] > 1)
                .max_by_key(|index| counts[*index])
                .ok_or(DatasetError::EmptyDataset)?;
            counts[index] -= 1;
        }

        let make_partition = |rows: &[usize]| {
            Dataset::new(
                rows.iter()
                    .map(|index| self.features[*index].clone())
                    .collect(),
                rows.iter()
                    .map(|index| self.targets[*index].clone())
                    .collect(),
            )
        };
        let train_end = counts[0];
        let validation_end = train_end + counts[1];
        Ok(DatasetSplit {
            train: make_partition(&indices[..train_end])?,
            validation: make_partition(&indices[train_end..validation_end])?,
            test: make_partition(&indices[validation_end..])?,
        })
    }
}

/// A mini-batch of supervised examples.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetBatch {
    features: Vec<Vector>,
    targets: Vec<Vector>,
}

impl DatasetBatch {
    /// Creates a dataset batch.
    pub fn new(features: Vec<Vector>, targets: Vec<Vector>) -> Self {
        Self { features, targets }
    }

    /// Returns the number of rows in this batch.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Returns true when this batch has no rows.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Returns batch features.
    pub fn features(&self) -> &[Vector] {
        &self.features
    }

    /// Returns batch targets.
    pub fn targets(&self) -> &[Vector] {
        &self.targets
    }
}

fn validate_dataset(features: &[Vector], targets: &[Vector]) -> Result<(), DatasetError> {
    if features.is_empty() || targets.is_empty() {
        return Err(DatasetError::EmptyDataset);
    }

    if features.len() != targets.len() {
        return Err(DatasetError::RowCountMismatch {
            features: features.len(),
            targets: targets.len(),
        });
    }

    if let Some(first) = features.first() {
        let expected = first.len();

        for (row, feature) in features.iter().enumerate() {
            if feature.len() != expected {
                return Err(DatasetError::FeatureDimensionMismatch {
                    row,
                    expected,
                    actual: feature.len(),
                });
            }
        }
    }

    for (row, values) in features.iter().chain(targets.iter()).enumerate() {
        for (column, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(DatasetError::NonFiniteValue {
                    row,
                    column,
                    value: *value,
                });
            }
        }
    }

    Ok(())
}

fn parse_csv_row(line: &str, row: usize) -> Result<Vec<f64>, DatasetError> {
    line.split(',')
        .enumerate()
        .map(|(column_index, value)| {
            let trimmed = value.trim();

            trimmed
                .parse::<f64>()
                .map_err(|_| DatasetError::ParseFloat {
                    row,
                    column: column_index + 1,
                    value: trimmed.to_string(),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_dataset_successfully() {
        let dataset = Dataset::new(
            vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])],
            vec![Vector::new(vec![0.0]), Vector::new(vec![1.0])],
        )
        .unwrap();

        assert_eq!(dataset.len(), 2);
        assert!(!dataset.is_empty());
        assert_eq!(dataset.feature_size(), 2);
        assert_eq!(dataset.target_size(), 1);
    }

    #[test]
    fn rejects_empty_dataset() {
        let dataset = Dataset::new(vec![], vec![]);

        assert_eq!(dataset, Err(DatasetError::EmptyDataset));
    }

    #[test]
    fn rejects_row_count_mismatch() {
        let dataset = Dataset::new(
            vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])],
            vec![Vector::new(vec![1.0])],
        );

        assert_eq!(
            dataset,
            Err(DatasetError::RowCountMismatch {
                features: 2,
                targets: 1
            })
        );
    }

    #[test]
    fn rejects_feature_dimension_mismatch() {
        let dataset = Dataset::new(
            vec![Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0])],
            vec![Vector::new(vec![1.0]), Vector::new(vec![0.0])],
        );

        assert_eq!(
            dataset,
            Err(DatasetError::FeatureDimensionMismatch {
                row: 1,
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn loads_dataset_from_csv_with_header() {
        let csv = "age,income,label\n25,50000,1\n40,80000,0\n";

        let dataset = Dataset::from_csv_str(csv, 1, true).unwrap();

        assert_eq!(
            dataset.features(),
            &[
                Vector::new(vec![25.0, 50000.0]),
                Vector::new(vec![40.0, 80000.0])
            ]
        );
        assert_eq!(
            dataset.targets(),
            &[Vector::new(vec![1.0]), Vector::new(vec![0.0])]
        );
    }

    #[test]
    fn loads_dataset_from_csv_without_header_and_multiple_targets() {
        let csv = "1,2,0,1\n3,4,1,0\n";

        let dataset = Dataset::from_csv_str(csv, 2, false).unwrap();

        assert_eq!(
            dataset.features(),
            &[Vector::new(vec![1.0, 2.0]), Vector::new(vec![3.0, 4.0])]
        );
        assert_eq!(
            dataset.targets(),
            &[Vector::new(vec![0.0, 1.0]), Vector::new(vec![1.0, 0.0])]
        );
    }

    #[test]
    fn rejects_empty_csv() {
        let dataset = Dataset::from_csv_str("feature,target\n", 1, true);

        assert_eq!(dataset, Err(DatasetError::EmptyCsv));
    }

    #[test]
    fn rejects_invalid_target_column_count() {
        let dataset = Dataset::from_csv_str("1,2,3\n", 3, false);

        assert_eq!(
            dataset,
            Err(DatasetError::InvalidTargetColumnCount {
                target_columns: 3,
                total_columns: 3
            })
        );
    }

    #[test]
    fn rejects_inconsistent_csv_column_counts() {
        let dataset = Dataset::from_csv_str("1,2,3\n4,5\n", 1, false);

        assert_eq!(
            dataset,
            Err(DatasetError::InconsistentColumnCount {
                row: 2,
                expected: 3,
                actual: 2
            })
        );
    }

    #[test]
    fn rejects_unparseable_csv_value() {
        let dataset = Dataset::from_csv_str("1,nope,0\n", 1, false);

        assert_eq!(
            dataset,
            Err(DatasetError::ParseFloat {
                row: 1,
                column: 2,
                value: "nope".to_string()
            })
        );
    }

    #[test]
    fn gets_row_by_index() {
        let dataset = Dataset::new(
            vec![Vector::new(vec![1.0, 2.0])],
            vec![Vector::new(vec![1.0])],
        )
        .unwrap();

        let (features, target) = dataset.get(0).unwrap();

        assert_eq!(features, &Vector::new(vec![1.0, 2.0]));
        assert_eq!(target, &Vector::new(vec![1.0]));
    }

    #[test]
    fn returns_error_for_invalid_row_index() {
        let dataset =
            Dataset::new(vec![Vector::new(vec![1.0])], vec![Vector::new(vec![0.0])]).unwrap();

        let result = dataset.get(1);

        assert_eq!(
            result,
            Err(DatasetError::IndexOutOfBounds {
                index: 1,
                length: 1
            })
        );
    }

    #[test]
    fn creates_batches() {
        let dataset = Dataset::new(
            vec![
                Vector::new(vec![1.0]),
                Vector::new(vec![2.0]),
                Vector::new(vec![3.0]),
            ],
            vec![
                Vector::new(vec![0.0]),
                Vector::new(vec![1.0]),
                Vector::new(vec![0.0]),
            ],
        )
        .unwrap();

        let batches = dataset.batches(2).unwrap();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
        assert_eq!(batches[1].features(), &[Vector::new(vec![3.0])]);
    }

    #[test]
    fn rejects_zero_batch_size() {
        let dataset =
            Dataset::new(vec![Vector::new(vec![1.0])], vec![Vector::new(vec![0.0])]).unwrap();

        let result = dataset.batches(0);

        assert_eq!(
            result,
            Err(DatasetError::InvalidBatchSize { batch_size: 0 })
        );
    }

    #[test]
    fn creates_train_test_split() {
        let dataset = Dataset::new(
            vec![
                Vector::new(vec![1.0]),
                Vector::new(vec![2.0]),
                Vector::new(vec![3.0]),
                Vector::new(vec![4.0]),
            ],
            vec![
                Vector::new(vec![0.0]),
                Vector::new(vec![1.0]),
                Vector::new(vec![0.0]),
                Vector::new(vec![1.0]),
            ],
        )
        .unwrap();

        let (train, test) = dataset.train_test_split(0.25).unwrap();

        assert_eq!(
            train.features(),
            &[
                Vector::new(vec![1.0]),
                Vector::new(vec![2.0]),
                Vector::new(vec![3.0])
            ]
        );
        assert_eq!(test.features(), &[Vector::new(vec![4.0])]);
    }

    #[test]
    fn rejects_invalid_train_test_ratio() {
        let dataset = Dataset::new(
            vec![Vector::new(vec![1.0]), Vector::new(vec![2.0])],
            vec![Vector::new(vec![0.0]), Vector::new(vec![1.0])],
        )
        .unwrap();

        let result = dataset.train_test_split(1.0);

        assert_eq!(result, Err(DatasetError::InvalidSplitRatio { ratio: 1.0 }));
    }

    #[test]
    fn produces_reproducible_three_way_splits() {
        let dataset = Dataset::new(
            (0..10)
                .map(|value| Vector::new(vec![value as f64]))
                .collect(),
            (0..10)
                .map(|value| Vector::new(vec![(value % 2) as f64]))
                .collect(),
        )
        .unwrap();
        let config = DatasetSplitConfig::new(0.7, 0.15, 0.15, 42).unwrap();
        let first = dataset.train_validation_test_split(config).unwrap();
        let second = dataset.train_validation_test_split(config).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.train().len() + first.validation().len() + first.test().len(),
            10
        );
        assert!(!first.train().is_empty());
        assert!(!first.validation().is_empty());
        assert!(!first.test().is_empty());
    }

    #[test]
    fn rejects_invalid_three_way_split_and_non_finite_data() {
        assert!(DatasetSplitConfig::new(0.7, 0.2, 0.2, 1).is_err());
        assert!(matches!(
            Dataset::new(
                vec![Vector::new(vec![f64::NAN])],
                vec![Vector::new(vec![0.0])]
            ),
            Err(DatasetError::NonFiniteValue { .. })
        ));
    }
}
