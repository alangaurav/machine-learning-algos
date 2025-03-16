import unittest
from mltools.matrix.matrix import Matrix
from mltools.regression.regression import Regression, LinearRegression, PolynomialRegression


class TestRegression(unittest.TestCase):

    def setUp(self):
        self.regression = Regression(learning_rate=0.01)
        self.linear_reg = LinearRegression()
        self.poly_reg = PolynomialRegression(degree=2)

    def test_gen_y_matrix(self):
        y = [1, 2, 3]
        y_matrix = self.regression.gen_y_matrix(y)
        self.assertEqual(y_matrix.matrix, [[1], [2], [3]])

    def test_fit_invalid_dimensions(self):
        x_train = [[1, 2], [3, 4]]
        y_train = [1, 2, 3]  # Incorrect dimension
        with self.assertRaises(TypeError):
            self.regression.fit(x_train, y_train)

    def test_predict_without_fit(self):
        x = [[1, 2], [3, 4]]
        with self.assertRaises(ValueError):
            self.linear_reg.predict(x)

    def test_gradient_descent_convergence(self):
        x = Matrix(3, 2, [1, 1, 1, 2, 2, 2])
        y = Matrix(3, 1, [1, 2, 3])
        self.linear_reg.gradient_descent(x, y)
        self.assertIsNotNone(self.linear_reg.theta)

    def test_linear_regression_fit_and_predict(self):
        x_train = [[1, 2], [2, 3], [3, 4]]
        y_train = [1, 2, 3]
        self.linear_reg.fit(x_train, y_train)
        predictions = self.linear_reg.predict(x_train)
        self.assertAlmostEqual(predictions[0], 1, delta=0.1)

    def test_polynomial_design_matrix(self):
        x = [1, 2, 3]
        design_matrix = self.poly_reg.gen_design_matrix(x)
        self.assertEqual(design_matrix.matrix, [[1, 1, 1], [1, 2, 4], [1, 3, 9]])

    def test_polynomial_regression_fit_and_predict(self):
        x_train = [1, 2, 3]
        y_train = [1, 4, 9]  # Quadratic relation
        self.poly_reg.fit(x_train, y_train)
        predictions = self.poly_reg.predict(x_train)
        self.assertAlmostEqual(predictions[0], 1, delta=0.1)


if __name__ == "__main__":
    unittest.main()
