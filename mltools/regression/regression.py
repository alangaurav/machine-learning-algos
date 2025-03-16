from mltools.matrix.matrix import Matrix
from mltools.metrics.metrics import Metrics
import random

class Regression:
    """
    A class to perform Regression using multiple variables.
    """

    def __init__(self, learning_rate):
        """
        Initializes the Regression model.
        """
        self.learning_rate = learning_rate
        self.theta = None

    def gen_y_matrix(self, y):
        """
        Generates the output matrix for the target values.

        Parameters:
        y (list): Target values.

        Returns:
        Matrix: The output matrix.
        """
        return Matrix(len(y), 1, y)

    def normal_form(self, x, y):
        """
        Computes the normal equation to find the optimal parameters (theta).

        Parameters:
        x (Matrix): The design matrix.
        y (Matrix): The output matrix.
        """
        # Transpose of the design matrix
        x_t = x.transpose()
        # X^T * X
        xtx = x_t.multiply(x)
        # Inverse of X^T * X
        xtx_inv = xtx.inverse()
        # X^T * Y
        xty = x_t.multiply(y)
        # Calculate theta
        self.theta = xtx_inv.multiply(xty)

    def convergence_condition(self, x, y, theta):
        """
        Checks for convergence based on Mean Squared Error (MSE).

        Parameters:
        x (Matrix): The design matrix.
        y (Matrix): The output matrix.
        theta (Matrix): Current parameters.

        Returns:
        bool: True if not converged, False if converged.
        """
        # Precision threshold
        prec = 0.001
        mets = Metrics()
        mse = mets.mean_squared_error(
            [ele for row in y.matrix for ele in row], self.predict(x, theta)
        )
        # Return True if MSE is above threshold
        return abs(mse) >= prec

    def gradient_descent(self, x, y):
        """
        Performs gradient descent to minimize the loss function.

        Parameters:
        x (Matrix): The design matrix.
        y (Matrix): The output matrix.
        """
        # Initialize weights
        theta = Matrix(len(x.matrix[0]), 1, [random.random()] * len(x.matrix[0]))
        # Intercept term
        intercept = theta.matrix[0][0]
        # Transpose of the design matrix
        x_t = x.transpose()
        # Learning rate
        gradient_neta = self.learning_rate
        intercept_neta = 0.01
        # Number of iterations
        epochs = 100000

        # Iterate until convergence or epoch limit
        for _ in range(epochs):
            y_preds = self.predict(x, theta)
            y_preds = self.gen_y_matrix(y_preds)
            y_dif = y_preds.subtract(y)
            gradient = x_t.multiply(y_dif)
            scale_factor = (2 * gradient_neta) / len(y.matrix)
            delta = gradient.scalarMultiply(scale_factor)
            theta = theta.subtract(delta)

            # Update intercept
            intercept_update = (2 / len(y.matrix)) * sum(
                [y_dif.matrix[i][0] for i in range(len(y.matrix))]
            )
            intercept -= intercept_update * intercept_neta
            theta.matrix[0][0] = intercept

            if not self.convergence_condition(x, y, theta):
                break

        self.theta = theta  # Final parameters

    def fit(self, x_train, y_train):
        """
        Fits the model to the training data.

        Parameters:
        x_train (list): Training features.
        y_train (list): Training target values.

        Raises:
        TypeError: If dimensions of X and Y do not match.
        """
        if len(y_train) != len(x_train):
            raise TypeError("Invalid dimensions of X and Y data.")
        x = self.gen_design_matrix(x_train)
        y = self.gen_y_matrix(y_train)
        self.gradient_descent(x, y)

    def predict(self, x, theta=None):
        """
        Makes predictions based on input features.

        Parameters:
        x (list): Input features.
        theta (Matrix): Parameters for prediction (optional).

        Returns:
        list: Predicted values.
        """
        if self.theta is None and theta is None:
            raise ValueError('Fit must be called!')
        if type(x) is list:
            x = self.gen_design_matrix(x)
        if theta is None:
            theta = self.theta
        preds = x.multiply(theta)
        return [ele for row in preds.matrix for ele in row]


# Multivariate Linear Regression, can be expanded to polynomial.
class LinearRegression(Regression):
    """
    A class to perform Linear Regression using multiple variables.
    """

    def __init__(self):
        """
        Initializes the Linear Regression model.
        """
        super().__init__(learning_rate=0.00001)

    def gen_design_matrix(self, x):
        """
        Generates the design matrix for the input features.

        Parameters:
        x (list): Input features, either a n-dimensional matrix or a flat list.

        Returns:
        Matrix: The design matrix with an added intercept term.
        """
        data = []
        for row in x:
            # Add intercept term
            data.append(1)
            if type(row) is int:
                data.append(row)
            else:
                data.extend(row)
        if type(x[0]) is int:
            design_mat = Matrix(len(x), 2, data)
        else:
            design_mat = Matrix(len(x), len(x[0]) + 1, data)

        return design_mat


class PolynomialRegression(Regression):
    """
    A class to perform Linear Regression using multiple variables.
    """

    def __init__(self, degree=2):
        super().__init__(learning_rate=0.000001)
        self.degree = degree

    def gen_design_matrix(self, x):
        """
        Generates the design matrix for the input features.

        Parameters:
        x (list): Input features, a flat list.

        Returns:
        Matrix: The design matrix with an added intercept term.
        """
        if type(x[0]) is int:
            feat = 1
        else:
            feat = len(x[0])    
        design_mat = Matrix(len(x), self.degree * feat + 1)
        for i in range(len(design_mat.matrix)):
            design_mat.matrix[i][0] = 1

        for i in range(len(design_mat.matrix)):
            if feat == 1:
                for k in range(1, self.degree + 1):
                    design_mat.matrix[i][k*feat] = x[i] ** k
            else:
                for k in range(1, self.degree + 1):
                    for j in range(feat):
                        design_mat.matrix[i][k*feat + j] = x[i][j] ** k
        return design_mat
        