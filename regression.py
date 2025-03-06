from matrix import Matrix
from metrics import Metrics
import random

# Multivariate Linear Regression, can be expanded to polynomial.
class LinearRegression:

    def __init__(self, degree=1):
        self.degree = degree

    def gen_x_matrix(self, x):
        data = []
        for row in x:
            if type(row) is int:
                data.append(1)
                data.append(row)
            else:
                data.append(1)
                for ele in row:
                    data.append(ele)
        return Matrix(len(x), len(x[0]) + 1, data)

    def gen_y_matrix(self, y):
        return Matrix(len(y), 1, y)

    def normal_form(self, x, y):
        x_t = x.transpose()
        # IMPORTANT: If there is a linear relationship between xi then this method is not possible.
        # i.e. Rank(X) >= m (if n x m matrix)
        xtx = x_t.multiply(x)
        xtx_inv = xtx.inverse()
        xty = x_t.multiply(y)
        self.theta = xtx_inv.multiply(xty)

    def fit(self, x_train, y_train):
        if len(y_train) != len(x_train):
            raise TypeError("Invalid dimensions of X and Y data.")
        x = self.gen_x_matrix(x_train)
        y = self.gen_y_matrix(y_train)
        self.gradient_descent(x, y)

    def predict(self, x, theta=None):
        if type(x) is list:
            x = self.gen_x_matrix(x)
        if theta is None:
            theta = self.theta
        preds = x.multiply(theta)
        return [ele for row in preds.matrix for ele in row]

    def convergence_condition(self, x, y, theta):
        prec = 0.001
        mets = Metrics()
        mse = mets.mean_squared_error(
            [ele for row in y.matrix for ele in row], self.predict(x, theta)
        )
        if abs(mse) < prec:
            return False
        return True

    # Gradient descent based on loss fn: MSE
    # This does not affect bias only weights
    # Bias needs to be checked separately
    def gradient_descent(self, x, y):
        theta = Matrix(
            len(x.matrix[0]),
            1,
            [random.random()] * len(x.matrix[0]),
        )
        intercept = theta.matrix[0][0]
        x_t = x.transpose()
        neta = 0.00001
        epochs = 1000
        counter = 0
        # Convergence is reached when prediction values do not change
        for _ in range(epochs):
            counter += 1
            y_preds = self.predict(x, theta)
            y_preds = self.gen_y_matrix(y_preds)
            y_dif = y_preds.subtract(y)
            # This is the MSE delta.
            gradient = x_t.multiply(y_dif)
            scale_factor = (2 * neta) / len(y.matrix)
            delta = gradient.scalarMultiply(scale_factor)
            theta = theta.subtract(delta)
            intercept_update = (2 / len(y.matrix)) * sum(
                [y_dif.matrix[i][0] for i in range(len(y.matrix))]
            )
            intercept -= intercept_update * 0.01
            theta.matrix[0][0] = intercept
            if not self.convergence_condition(x, y, theta):
                break

        self.theta = theta
