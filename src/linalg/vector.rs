use crate::linalg::errors::VectorError;

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
    pub fn get(&self, index: usize) -> Result<f64, VectorError> {
        self.values
            .get(index)
            .copied()
            .ok_or(VectorError::IndexOutOfBounds {
                index,
                length: self.len(),
            })
    }

    /// Adds two vectors.
    pub fn add(&self, other: &Vector) -> Result<Vector, VectorError> {
        if self.len() != other.len() {
            return Err(VectorError::DimensionMismatch {
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
}
