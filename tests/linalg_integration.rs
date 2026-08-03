use dcgaimodel::linalg::matrix::Matrix;

#[test]
fn transpose_of_identity_is_identity() {
    let identity = Matrix::identity(3);

    assert_eq!(identity.transpose(), identity);
}

#[test]
fn identity_preserves_matrix() {
    let matrix = Matrix::new(vec![vec![2.0, 3.0], vec![4.0, 5.0]]).unwrap();

    let identity = Matrix::identity(2);

    let result = identity.multiply_matrix(&matrix).unwrap();

    assert_eq!(result, matrix);
}
