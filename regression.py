from matrix import Matrix
from metrics import Metrics
import random

# Multivariate Linear Regression, can be expanded to polynomial.
class LinearRegression:

    def __init__(self, degree=1):
        self.degree = degree

    def gen_x_matrix(self, x):
        data = []
        for i in range(len(x)):
            row = x[i]
            if type(x[i]) is int:
                x[i] = [1, x[i]]
            else:
                row.insert(0, 1)
            data.extend(x[i])
        return Matrix(len(x), len(x[0]), data)

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
        prec = 0.01
        mets = Metrics()
        mse = mets.mean_squared_error(
            [ele for row in y.matrix for ele in row], self.predict(x, theta)
        )
        if abs(mse) < prec:
            return False
        return True

    # Gradient descent based on loss fn: MSE
    # TODO: Fix constant value rescaling.
    def gradient_descent(self, x, y):
        x_t = x.transpose()
        theta = Matrix(
            len(x.matrix[0]),
            1,
            [0] * len(x.matrix[0]),
        )
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
            # scale_factor = (2 / len(y.matrix)) * neta
            scale_factor = (2 * neta) / len(y.matrix)
            delta = gradient.scalarMultiply(scale_factor)
            theta = theta.subtract(delta)
            if not self.convergence_condition(x, y_preds, theta):
                break

        self.theta = theta


if __name__ == "__main__":
    # Let the function be y = 12x + 7z - 11
    x_train = [[random.randint(-100, 100), random.randint(50, 300)] for _ in range(10)]
    y_train = [12 * x_train[i][0] + 7 * x_train[i][1] - 11 for i in range(10)]
    model = LinearRegression()
    model.fit(x_train, y_train)
    x_test = [[random.randint(-10, 10), random.randint(5, 30)] for _ in range(10)]
    y_test = [12 * x_test[i][0] + 7 * x_test[i][1] - 11 for i in range(10)]
    y_preds = model.predict(x_test)
    mets = Metrics()
    mse = mets.mean_squared_error(y_test, y_preds)
    print(mse)
    print(model.theta.matrix)
