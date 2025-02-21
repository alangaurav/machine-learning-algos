from matrix import Matrix
import unittest


class TestMatrix(unittest.TestCase):

    def setUp(self):
        # Setting up some basic test matrices for reuse in tests
        self.matrix1 = Matrix(2, 2, [1, 2, 3, 4])  # Matrix: [[1, 2], [3, 4]]
        self.matrix2 = Matrix(2, 2, [5, 6, 7, 8])  # Matrix: [[5, 6], [7, 8]]
        self.matrix3 = Matrix(
            3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]
        )  # Matrix: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.matrix4 = Matrix(
            3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1]
        )  # Identity Matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def test_multiply_valid(self):
        # Test matrix multiplication for valid matrices
        result = self.matrix1.multiply(self.matrix2)
        self.assertEqual(result.matrix, [[19, 22], [43, 50]])

    def test_multiply_incompatible(self):
        # Test matrix multiplication for incompatible matrices
        matrix_invalid = Matrix(3, 2, [1, 2, 3, 4, 5, 6])
        with self.assertRaises(ValueError):
            self.matrix1.multiply(matrix_invalid)

    def test_scalar_multiply(self):
        # Test scalar multiplication
        result = self.matrix1.scalarMultiply(2)
        self.assertEqual(result.matrix, [[2, 4], [6, 8]])

    def test_transpose(self):
        # Test transpose of a matrix
        result = self.matrix1.transpose()
        self.assertEqual(result.matrix, [[1, 3], [2, 4]])

    def test_determinant_2x2(self):
        # Test determinant of 2x2 matrix
        result = self.matrix1.determinant()
        self.assertEqual(result, -2)

    def test_determinant_3x3(self):
        # Test determinant of 3x3 matrix
        result = self.matrix3.determinant()
        self.assertEqual(result, 0)

    def test_determinant_non_square(self):
        # Test non-square matrix determinant (should raise an error)
        matrix_non_square = Matrix(2, 3, [1, 2, 3, 4, 5, 6])
        with self.assertRaises(ValueError):
            matrix_non_square.determinant()

    def test_inverse_valid(self):
        # Test inverse of a valid invertible matrix
        result = self.matrix4.inverse()
        self.assertEqual(
            result.matrix, self.matrix4.matrix
        )  # Identity matrix should be its own inverse

    def test_inverse_invalid(self):
        # Test inverse of a matrix that is not invertible
        matrix_singular = Matrix(2, 2, [1, 2, 2, 4])  # Singular matrix (det = 0)
        with self.assertRaises(ValueError):
            matrix_singular.inverse()

    def test_minor(self):
        # Test minor of a matrix
        result = self.matrix3.minor(self.matrix3.matrix)
        self.assertEqual(result.matrix, [[-3, -6, -3], [-6, -12, -6], [-3, -6, -3]])

    def test_cofactor(self):
        # Test cofactor of a matrix
        result = self.matrix3.cofactor(self.matrix3.matrix)
        self.assertEqual(result.matrix, [[-3, 6, -3], [6, -12, 6], [-3, 6, -3]])

    def test_adjunct(self):
        # Test adjunct (adjugate) of a matrix
        result = self.matrix3.adjunct(self.matrix3.matrix)
        self.assertEqual(result.matrix, [[-3, 6, -3], [6, -12, 6], [-3, 6, -3]])

    def test_sub_mat(self):
        # Test submatrix extraction
        result = self.matrix3.sub_mat(self.matrix3.matrix, 0, 0)
        self.assertEqual(result, [[5, 6], [8, 9]])

    def test_empty_matrix(self):
        # Test initialization of an empty matrix
        matrix_empty = Matrix()
        self.assertIsNone(matrix_empty.matrix)

    def test_invalid_matrix_multiply(self):
        # Test invalid matrix multiplication (with non-matching dimensions)
        matrix_invalid = Matrix(3, 2, [1, 2, 3, 4, 5, 6])
        with self.assertRaises(ValueError):
            self.matrix1.multiply(matrix_invalid)


if __name__ == "__main__":
    unittest.main()
