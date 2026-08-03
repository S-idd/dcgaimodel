use crate::dataset::{Dataset, DatasetError};
use crate::linalg::Vector;

/// Feature-wise standard-score normalization statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct StandardScaler {
    means: Vector,
    standard_deviations: Vector,
}

impl StandardScaler {
    /// Fits a standard scaler from dataset features.
    pub fn fit(dataset: &Dataset) -> Result<Self, DatasetError> {
        let feature_size = dataset.feature_size();

        if feature_size == 0 {
            return Err(DatasetError::EmptyDataset);
        }

        let mut means = vec![0.0; feature_size];

        for features in dataset.features() {
            for (index, value) in features.iter().enumerate() {
                means[index] += value;
            }
        }

        for mean in &mut means {
            *mean /= dataset.len() as f64;
        }

        let mut variances = vec![0.0; feature_size];

        for features in dataset.features() {
            for (index, value) in features.iter().enumerate() {
                let difference = value - means[index];
                variances[index] += difference * difference;
            }
        }

        let standard_deviations = variances
            .iter()
            .map(|variance| {
                let standard_deviation = (variance / dataset.len() as f64).sqrt();

                if standard_deviation == 0.0 {
                    1.0
                } else {
                    standard_deviation
                }
            })
            .collect();

        Ok(Self {
            means: Vector::new(means),
            standard_deviations: Vector::new(standard_deviations),
        })
    }

    /// Returns feature means.
    pub fn means(&self) -> &Vector {
        &self.means
    }

    /// Returns feature standard deviations.
    pub fn standard_deviations(&self) -> &Vector {
        &self.standard_deviations
    }

    /// Transforms one feature vector.
    pub fn transform_vector(&self, features: &Vector) -> Result<Vector, DatasetError> {
        if features.len() != self.means.len() {
            return Err(DatasetError::FeatureDimensionMismatch {
                row: 0,
                expected: self.means.len(),
                actual: features.len(),
            });
        }

        let values = features
            .iter()
            .zip(self.means.iter())
            .zip(self.standard_deviations.iter())
            .map(|((value, mean), standard_deviation)| (value - mean) / standard_deviation)
            .collect();

        Ok(Vector::new(values))
    }

    /// Transforms all feature vectors in a dataset while preserving targets.
    pub fn transform_dataset(&self, dataset: &Dataset) -> Result<Dataset, DatasetError> {
        let features = dataset
            .features()
            .iter()
            .map(|features| self.transform_vector(features))
            .collect::<Result<Vec<Vector>, DatasetError>>()?;

        Dataset::new(features, dataset.targets().to_vec())
    }

    /// Fits a scaler and returns the normalized dataset.
    pub fn fit_transform(dataset: &Dataset) -> Result<(Dataset, Self), DatasetError> {
        let scaler = Self::fit(dataset)?;
        let normalized = scaler.transform_dataset(dataset)?;

        Ok((normalized, scaler))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-10, "{left} != {right}");
    }

    fn dataset() -> Dataset {
        Dataset::new(
            vec![
                Vector::new(vec![1.0, 10.0]),
                Vector::new(vec![2.0, 10.0]),
                Vector::new(vec![3.0, 10.0]),
            ],
            vec![
                Vector::new(vec![0.0]),
                Vector::new(vec![1.0]),
                Vector::new(vec![0.0]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn fits_standard_scaler() {
        let scaler = StandardScaler::fit(&dataset()).unwrap();

        assert_eq!(scaler.means(), &Vector::new(vec![2.0, 10.0]));
        assert_close(scaler.standard_deviations()[0], (2.0_f64 / 3.0).sqrt());
        assert_eq!(scaler.standard_deviations()[1], 1.0);
    }

    #[test]
    fn transforms_feature_vector() {
        let scaler = StandardScaler::fit(&dataset()).unwrap();

        let transformed = scaler
            .transform_vector(&Vector::new(vec![3.0, 10.0]))
            .unwrap();

        assert_close(transformed[0], 1.0 / (2.0_f64 / 3.0).sqrt());
        assert_eq!(transformed[1], 0.0);
    }

    #[test]
    fn transform_rejects_dimension_mismatch() {
        let scaler = StandardScaler::fit(&dataset()).unwrap();

        let result = scaler.transform_vector(&Vector::new(vec![1.0]));

        assert_eq!(
            result,
            Err(DatasetError::FeatureDimensionMismatch {
                row: 0,
                expected: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn fit_transform_normalizes_dataset_features() {
        let (normalized, scaler) = StandardScaler::fit_transform(&dataset()).unwrap();

        assert_eq!(scaler.means(), &Vector::new(vec![2.0, 10.0]));
        assert_eq!(normalized.targets(), dataset().targets());
        assert_close(normalized.features()[0][0], -1.0 / (2.0_f64 / 3.0).sqrt());
        assert_eq!(normalized.features()[1][0], 0.0);
        assert_close(normalized.features()[2][0], 1.0 / (2.0_f64 / 3.0).sqrt());
        assert_eq!(normalized.features()[0][1], 0.0);
    }
}
