use crate::linalg::errors::LinalgError;
use std::fmt;
/// Represents a mathematical matrix.
///
/// The matrix is stored in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    values: Vec<Vec<f64>>,
}

impl Matrix {
    /// Creates a new matrix.
    pub fn new(values: Vec<Vec<f64>>) -> Result<Self, LinalgError> {
        if values.is_empty() {
            return Ok(Self { values });
        }

        let cols = values[0].len();

        if values.iter().any(|row| row.len() != cols) {
            return Err(LinalgError::DimensionMismatch {
                left: cols,
                right: 0,
            });
        }

        Ok(Self { values })
    }

    /// Returns the number of rows.
    pub fn rows(&self) -> usize {
        self.values.len()
    }

    /// Returns the number of columns.
    pub fn cols(&self) -> usize {
        if self.values.is_empty() {
            0
        } else {
            self.values[0].len()
        }
    }

    /// Returns the shape of the matrix.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows(), self.cols())
    }

    /// Returns the element at the specified row and column.
    pub fn get(&self, row: usize, col: usize) -> Result<f64, LinalgError> {
        if row >= self.rows() {
            return Err(LinalgError::IndexOutOfBounds {
                index: row,
                length: self.rows(),
            });
        }

        if col >= self.cols() {
            return Err(LinalgError::IndexOutOfBounds {
                index: col,
                length: self.cols(),
            });
        }

        Ok(self.values[row][col])
    }

    /// Adds two matrices.
    ///
    /// Returns an error if the matrix dimensions do not match.
    pub fn add(&self, other: &Matrix) -> Result<Matrix, LinalgError> {
        if self.shape() != other.shape() {
            return Err(LinalgError::DimensionMismatch {
                left: self.rows(),
                right: other.rows(),
            });
        }

        let values = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(row_a, row_b)| row_a.iter().zip(row_b.iter()).map(|(a, b)| a + b).collect())
            .collect();

        Ok(Matrix::new(values).unwrap())
    }

    /// Subtracts another matrix from this matrix.
    ///
    /// Returns an error if the matrix dimensions do not match.
    pub fn subtract(&self, other: &Matrix) -> Result<Matrix, LinalgError> {
        if self.shape() != other.shape() {
            return Err(LinalgError::DimensionMismatch {
                left: self.rows(),
                right: other.rows(),
            });
        }

        let values = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(row_a, row_b)| row_a.iter().zip(row_b.iter()).map(|(a, b)| a - b).collect())
            .collect();

        Ok(Matrix::new(values).unwrap())
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "[")?;

        for row in &self.values {
            write!(f, "  [")?;

            for (index, value) in row.iter().enumerate() {
                if index > 0 {
                    write!(f, ", ")?;
                }

                write!(f, "{value}")?;
            }

            writeln!(f, "]")?;
        }

        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_matrix_successfully() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        assert_eq!(matrix.rows(), 2);
        assert_eq!(matrix.cols(), 2);
        assert_eq!(matrix.shape(), (2, 2));
    }

    #[test]
    fn creates_empty_matrix() {
        let matrix = Matrix::new(vec![]).unwrap();

        assert_eq!(matrix.shape(), (0, 0));
    }

    #[test]
    fn rejects_irregular_matrix() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0]]);

        assert!(matrix.is_err());
    }

    #[test]
    fn gets_valid_matrix_element() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        assert_eq!(matrix.get(1, 0).unwrap(), 3.0);
    }

    #[test]
    fn returns_error_for_invalid_row() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        assert!(matrix.get(2, 0).is_err());
    }

    #[test]
    fn returns_error_for_invalid_column() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        assert!(matrix.get(0, 2).is_err());
    }

    #[test]
    fn displays_matrix_correctly() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let expected = "[\n  [1, 2]\n  [3, 4]\n]";

        assert_eq!(format!("{}", matrix), expected);
    }

    #[test]
    fn adds_two_matrices() {
        let a = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let b = Matrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

        let result = a.add(&b).unwrap();

        let expected = Matrix::new(vec![vec![6.0, 8.0], vec![10.0, 12.0]]).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn matrix_add_dimension_mismatch_returns_error() {
        let a = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let b = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();

        assert!(a.add(&b).is_err());
    }

    #[test]
    fn adds_empty_matrices() {
        let a = Matrix::new(vec![]).unwrap();
        let b = Matrix::new(vec![]).unwrap();

        let result = a.add(&b).unwrap();

        assert_eq!(result.shape(), (0, 0));
    }

    #[test]
    fn subtracts_two_matrices() {
        let a = Matrix::new(vec![vec![5.0, 6.0], vec![7.0, 8.0]]).unwrap();

        let b = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let result = a.subtract(&b).unwrap();

        let expected = Matrix::new(vec![vec![4.0, 4.0], vec![4.0, 4.0]]).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn matrix_subtract_dimension_mismatch_returns_error() {
        let a = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let b = Matrix::new(vec![vec![1.0], vec![2.0]]).unwrap();

        assert!(a.subtract(&b).is_err());
    }

    #[test]
    fn subtracts_empty_matrices() {
        let a = Matrix::new(vec![]).unwrap();
        let b = Matrix::new(vec![]).unwrap();

        let result = a.subtract(&b).unwrap();

        assert_eq!(result.shape(), (0, 0));
    }
}
