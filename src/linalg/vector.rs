use crate::linalg::errors::LinalgError;
use std::fmt;
use std::ops::Index;
use std::slice::Iter;
/// Represents a mathematical vector.
///
/// A vector is an ordered collection of floating-point values.
/// It is the fundamental data structure used throughout our
/// AI engine for representing inputs, outputs, weights, and biases.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    values: Vec<f64>,
}

impl Vector {
    /// Creates a new vector.
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns the value at the given index.
    pub fn get(&self, index: usize) -> Result<f64, LinalgError> {
        self.values
            .get(index)
            .copied()
            .ok_or(LinalgError::IndexOutOfBounds {
                index,
                length: self.len(),
            })
    }

    /// Adds two vectors.
    pub fn add(&self, other: &Vector) -> Result<Vector, LinalgError> {
        if self.len() != other.len() {
            return Err(LinalgError::DimensionMismatch {
                left: self.len(),
                right: other.len(),
            });
        }

        let values = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Vector::new(values))
    }

    /// Subtracts another vector from this vector.
    pub fn subtract(&self, other: &Vector) -> Result<Vector, LinalgError> {
        if self.len() != other.len() {
            return Err(LinalgError::DimensionMismatch {
                left: self.len(),
                right: other.len(),
            });
        }

        let values = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a - b)
            .collect();

        Ok(Vector::new(values))
    }

    /// Multiplies every element in the vector by a scalar.
    pub fn scalar_multiply(&self, scalar: f64) -> Vector {
        let values = self.values.iter().map(|value| value * scalar).collect();

        Vector::new(values)
    }

    /// Computes the dot product of two vectors.
    ///
    /// Returns an error if the vectors have different dimensions.
    pub fn dot_product(&self, other: &Vector) -> Result<f64, LinalgError> {
        if self.len() != other.len() {
            return Err(LinalgError::DimensionMismatch {
                left: self.len(),
                right: other.len(),
            });
        }

        let result = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();

        Ok(result)
    }

    /// Computes the Euclidean magnitude (L2 norm) of the vector.
    pub fn magnitude(&self) -> f64 {
        self.values
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    }

    /// Returns a normalized version of the vector.
    ///
    /// A normalized vector has a magnitude of 1.
    pub fn normalize(&self) -> Result<Vector, LinalgError> {
        let magnitude = self.magnitude();

        if magnitude == 0.0 {
            return Err(LinalgError::ZeroMagnitude);
        }

        let values = self.values.iter().map(|value| value / magnitude).collect();

        Ok(Vector::new(values))
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;

        for (index, value) in self.values.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }

            write!(f, "{value}")?;
        }

        write!(f, "]")
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<'a> IntoIterator for &'a Vector {
    type Item = &'a f64;
    type IntoIter = Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_vector_successfully() {
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        assert_eq!(vector.len(), 3);
        assert!(!vector.is_empty());
    }

    #[test]
    fn creates_empty_vector() {
        let vector = Vector::new(vec![]);

        assert_eq!(vector.len(), 0);
        assert!(vector.is_empty());
    }

    #[test]
    fn gets_valid_element() {
        let vector = Vector::new(vec![10.0, 20.0, 30.0]);

        assert_eq!(vector.get(1).unwrap(), 20.0);
    }

    #[test]
    fn returns_error_for_invalid_index() {
        let vector = Vector::new(vec![1.0, 2.0]);

        assert!(vector.get(5).is_err());
    }

    #[test]
    fn adds_two_vectors() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);

        let result = a.add(&b).unwrap();

        assert_eq!(result, Vector::new(vec![5.0, 7.0, 9.0]));
    }

    #[test]
    fn add_dimension_mismatch_returns_error() {
        let a = Vector::new(vec![1.0, 2.0]);
        let b = Vector::new(vec![3.0, 4.0, 5.0]);

        assert!(a.add(&b).is_err());
    }

    #[test]
    fn subtracts_two_vectors() {
        let a = Vector::new(vec![5.0, 7.0, 9.0]);
        let b = Vector::new(vec![1.0, 2.0, 3.0]);

        let result = a.subtract(&b).unwrap();

        assert_eq!(result, Vector::new(vec![4.0, 5.0, 6.0]));
    }

    #[test]
    fn subtract_dimension_mismatch_returns_error() {
        let a = Vector::new(vec![1.0, 2.0]);
        let b = Vector::new(vec![1.0, 2.0, 3.0]);

        assert!(a.subtract(&b).is_err());
    }

    #[test]
    fn multiplies_vector_by_positive_scalar() {
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        let result = vector.scalar_multiply(2.0);

        assert_eq!(result, Vector::new(vec![2.0, 4.0, 6.0]));
    }

    #[test]
    fn multiplies_vector_by_zero() {
        let vector = Vector::new(vec![5.0, 10.0, 15.0]);

        let result = vector.scalar_multiply(0.0);

        assert_eq!(result, Vector::new(vec![0.0, 0.0, 0.0]));
    }

    #[test]
    fn multiplies_vector_by_negative_scalar() {
        let vector = Vector::new(vec![1.0, -2.0, 3.0]);

        let result = vector.scalar_multiply(-2.0);

        assert_eq!(result, Vector::new(vec![-2.0, 4.0, -6.0]));
    }

    #[test]
    fn computes_dot_product() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![4.0, 5.0, 6.0]);

        let result = a.dot_product(&b).unwrap();

        assert_eq!(result, 32.0);
    }

    #[test]
    fn dot_product_dimension_mismatch_returns_error() {
        let a = Vector::new(vec![1.0, 2.0]);
        let b = Vector::new(vec![3.0, 4.0, 5.0]);

        assert!(a.dot_product(&b).is_err());
    }

    #[test]
    fn dot_product_with_zero_vector() {
        let a = Vector::new(vec![1.0, 2.0, 3.0]);
        let b = Vector::new(vec![0.0, 0.0, 0.0]);

        let result = a.dot_product(&b).unwrap();

        assert_eq!(result, 0.0);
    }

    #[test]
    fn computes_vector_magnitude() {
        let vector = Vector::new(vec![3.0, 4.0]);

        assert_eq!(vector.magnitude(), 5.0);
    }

    #[test]
    fn magnitude_of_zero_vector() {
        let vector = Vector::new(vec![0.0, 0.0, 0.0]);

        assert_eq!(vector.magnitude(), 0.0);
    }

    #[test]
    fn magnitude_of_single_element_vector() {
        let vector = Vector::new(vec![7.0]);

        assert_eq!(vector.magnitude(), 7.0);
    }

    #[test]
    fn normalizes_vector() {
        let vector = Vector::new(vec![3.0, 4.0]);

        let normalized = vector.normalize().unwrap();

        assert!((normalized.magnitude() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_zero_vector_returns_error() {
        let vector = Vector::new(vec![0.0, 0.0, 0.0]);

        assert!(vector.normalize().is_err());
    }

    #[test]
    fn displays_vector_correctly() {
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        assert_eq!(format!("{}", vector), "[1, 2, 3]");
    }

    #[test]
    fn indexes_first_element() {
        let vector = Vector::new(vec![10.0, 20.0, 30.0]);

        assert_eq!(vector[0], 10.0);
    }

    #[test]
    fn indexes_last_element() {
        let vector = Vector::new(vec![10.0, 20.0, 30.0]);

        assert_eq!(vector[2], 30.0);
    }

    #[test]
    fn iterates_over_vector() {
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        let collected: Vec<f64> = (&vector).into_iter().copied().collect();

        assert_eq!(collected, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn sums_elements_using_iterator() {
        let vector = Vector::new(vec![1.0, 2.0, 3.0]);

        let sum: f64 = (&vector).into_iter().copied().sum();

        assert_eq!(sum, 6.0);
    }
}
