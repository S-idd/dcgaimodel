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
        self.element_wise(other, |a, b| a + b)
    }

    /// Subtracts another matrix from this matrix.
    ///
    /// Returns an error if the matrix dimensions do not match.
    pub fn subtract(&self, other: &Matrix) -> Result<Matrix, LinalgError> {
        self.element_wise(other, |a, b| a - b)
    }

    fn element_wise<F>(&self, other: &Matrix, op: F) -> Result<Matrix, LinalgError>
    where
        F: Fn(f64, f64) -> f64,
    {
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
            .map(|(row_a, row_b)| {
                row_a
                    .iter()
                    .zip(row_b.iter())
                    .map(|(&a, &b)| op(a, b))
                    .collect()
            })
            .collect();

        Ok(Matrix::new(values).expect("element-wise operations always produce a valid matrix"))
    }

    /// Multiplies every element of the matrix by a scalar.
    pub fn scalar_multiply(&self, scalar: f64) -> Matrix {
        let values = self
            .values
            .iter()
            .map(|row| row.iter().map(|value| value * scalar).collect::<Vec<f64>>())
            .collect::<Vec<Vec<f64>>>();

        Matrix::new(values).expect("Scalar multiplication always produces a valid matrix")
    }

    /// Returns the transpose of the matrix.
    pub fn transpose(&self) -> Matrix {
        if self.rows() == 0 || self.cols() == 0 {
            return Matrix::new(vec![]).expect("An empty matrix is always valid");
        }

        let mut values = vec![vec![0.0; self.rows()]; self.cols()];

        for row in 0..self.rows() {
            for col in 0..self.cols() {
                values[col][row] = self.values[row][col];
            }
        }

        Matrix::new(values).expect("Transpose always produces a valid matrix")
    }

    /// Creates an identity matrix of the given size.
    pub fn identity(size: usize) -> Matrix {
        let mut values = vec![vec![0.0; size]; size];

        for i in 0..size {
            values[i][i] = 1.0;
        }

        Matrix::new(values).expect("Identity matrix is always valid")
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

    #[test]
    fn multiplies_matrix_by_positive_scalar() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let result = matrix.scalar_multiply(2.0);

        let expected = Matrix::new(vec![vec![2.0, 4.0], vec![6.0, 8.0]]).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn multiplies_matrix_by_zero() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let result = matrix.scalar_multiply(0.0);

        let expected = Matrix::new(vec![vec![0.0, 0.0], vec![0.0, 0.0]]).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn multiplies_matrix_by_negative_scalar() {
        let matrix = Matrix::new(vec![vec![1.0, -2.0], vec![3.0, -4.0]]).unwrap();

        let result = matrix.scalar_multiply(-2.0);

        let expected = Matrix::new(vec![vec![-2.0, 4.0], vec![-6.0, 8.0]]).unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn transposes_square_matrix() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

        let expected = Matrix::new(vec![vec![1.0, 3.0], vec![2.0, 4.0]]).unwrap();

        assert_eq!(matrix.transpose(), expected);
    }

    #[test]
    fn transposes_rectangular_matrix() {
        let matrix = Matrix::new(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]).unwrap();

        let expected = Matrix::new(vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]).unwrap();

        assert_eq!(matrix.transpose(), expected);
    }

    #[test]
    fn transpose_of_empty_matrix_is_empty() {
        let matrix = Matrix::new(vec![]).unwrap();

        assert_eq!(matrix.transpose().shape(), (0, 0));
    }

    #[test]
    fn creates_identity_matrix_1x1() {
        let matrix = Matrix::identity(1);

        let expected = Matrix::new(vec![vec![1.0]]).unwrap();

        assert_eq!(matrix, expected);
    }

    #[test]
    fn creates_identity_matrix_3x3() {
        let matrix = Matrix::identity(3);

        let expected = Matrix::new(vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ])
        .unwrap();

        assert_eq!(matrix, expected);
    }

    #[test]
    fn creates_identity_matrix_0x0() {
        let matrix = Matrix::identity(0);

        assert_eq!(matrix.shape(), (0, 0));
    }
}
