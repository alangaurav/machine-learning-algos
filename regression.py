from matrix import Matrix
from metrics import Metrics
import random


# Multivariate Linear Regression, can be expanded to polynomial.
class LinearRegression:
    """
    A class to perform Linear Regression using multiple variables.
    """

    def __init__(self):
        """
        Initializes the LinearRegression model.
        """
        self.theta = None

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
        return Matrix(len(x), len(x[0]) + 1, data)

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
        gradient_neta = 0.00001
        intercept_neta = 0.01
        # Number of iterations
        epochs = 1000
        counter = 0

        # Iterate until convergence or epoch limit
        for _ in range(epochs):
            counter += 1
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
        if type(x) is list:
            x = self.gen_design_matrix(x)
        if theta is None:
            theta = self.theta
        preds = x.multiply(theta)
        return [ele for row in preds.matrix for ele in row]
