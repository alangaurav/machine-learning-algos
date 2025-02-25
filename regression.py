from matrix import Matrix
from metrics import Metrics
import random

# Multivariate Linear Regression, can be expanded to polynomial.
class LinearRegression:
    def __init__(self, degree = 1):
        self.degree = degree

    def fit(self, x_train, y_train):
        if len(y_train) != len(x_train):
            raise TypeError("Invalid dimensions of X and Y data.")

        y = Matrix(len(y_train), 1, y_train)
        data = []
        for i in range(len(x_train)):
            row = x_train[i]
            if type(x_train[i]) is int:
                x_train[i] = [1, x_train[i]]
            else:
                row.insert(0, 1)
            data.extend(x_train[i])

        x = Matrix(len(x_train), len(x_train[0]), data)
        x_t = x.transpose()
        # Change this to the summation fn. matrix for XtX * XtY => summation of xi ** i matrix * summation of yi ** i matrix
        # IMPORTANT: If there is a linear relationship between xi then this method is not possible.
        # i.e. Rank(X) >= m (if n x m matrix)
        xtx = x_t.multiply(x)
        xtx_inv = xtx.inverse()
        xty = x_t.multiply(y)
        self.theta = xtx_inv.multiply(xty)
        self.x = x

    def predict(self, x, theta = None):
        data = []
        for i in range(len(x)):
            row = x[i]
            if type(x[i]) is int:
                x[i] = [1, x[i]]
            else:
                row.insert(0, 1)
            data.extend(x[i])

        x = Matrix(len(x), len(x[0]), data)

        if theta is None:
            theta = self.theta
        preds = x.multiply(theta)
        return [ele for row in preds.matrix for ele in row]

    # Gradient descent based on loss fn: MSE
    # Gradient descent should be based on known y values.
    # How do you achieve convergence??
    def gradient_descent(self, y):
        x_t = self.x.transpose()
        theta = self.theta
        y_preds_new = self.predict(self.x)
        y_preds_old = Matrix(len(y_preds_new.matrix), 1, y)
        i = 0
        # Convergence is reached when prediction values do not change
        while i < 20 and not y_preds_new.equal(y_preds_old):
            y_dif = y_preds_new.subtract(y_preds_old)
            xtydif = x_t.multiply(y_dif)
            learn_rate = xtydif.scalarMultiply((2) / len(self.theta.matrix))
            theta = theta.subtract(learn_rate)
            print(theta.matrix)
            y_preds_old = y_preds_new
            y_preds_new = self.predict(self.x, theta)
            i += 1

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
    print(round(mse, 4))
