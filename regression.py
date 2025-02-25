from matrix import Matrix

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
