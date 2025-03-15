class Matrix:
    """
    A class to represent and perform operations on matrices.
    """

    def __init__(self, n=None, m=None, data=None):
        """
        Initializes a matrix with given dimensions and optional data.

        Parameters:
        n (int): Number of rows.
        m (int): Number of columns.
        data (list): Optional flat list of matrix elements.
        """
        if n == None or m == None:
            self.matrix = None
            return
        elif data is not None:
            matrix = [[data[i * (m - 1) + j + i] for j in range(m)] for i in range(n)]
        else:
            matrix = [[0 for _ in range(m)] for _ in range(n)]
        self.matrix = matrix

    def add(self, matrix):
        """
        Addition of another matrix to the current matrix.

        Parameters:
        matrix (Matrix): The matrix to be added.

        Returns:
        Matrix: The resulting matrix after addition.

        Raises:
        ValueError: If matrices have incompatible dimensions.
        """
        if len(self.matrix) != len(matrix.matrix) or len(self.matrix[0]) != len(
            matrix.matrix[0]
        ):
            raise ValueError("Incompatible matrices!")
        res = Matrix(len(self.matrix), len(self.matrix[0]))
        res.matrix = [
            [
                self.matrix[i][j] + matrix.matrix[i][j]
                for j in range(len(self.matrix[0]))
            ]
            for i in range(len(self.matrix))
        ]
        return res

    def subtract(self, matrix):
        """
        Subtraction of another matrix from the current matrix.

        Parameters:
        matrix (Matrix): The matrix to be subtracted.

        Returns:
        Matrix: The resulting matrix after subtraction.

        Raises:
        ValueError: If matrices have incompatible dimensions.
        """
        if len(self.matrix) != len(matrix.matrix) or len(self.matrix[0]) != len(
            matrix.matrix[0]
        ):
            raise ValueError("Incompatible matrices!")
        res = Matrix(len(self.matrix), len(self.matrix[0]))
        res.matrix = [
            [
                self.matrix[i][j] - matrix.matrix[i][j]
                for j in range(len(self.matrix[0]))
            ]
            for i in range(len(self.matrix))
        ]
        return res

    def multiply(self, matrix):
        """
        Multiplies the current matrix by another matrix.

        Parameters:
        matrix (Matrix): The matrix to multiply with.

        Returns:
        Matrix: The resulting matrix after multiplication.

        Raises:
        ValueError: If matrices have incompatible dimensions.
        """
        matrix = matrix.matrix
        if len(self.matrix[0]) != len(matrix):
            raise ValueError("Incompatible matrices")

        result = Matrix(
            len(self.matrix), len(matrix[0]), [0] * len(self.matrix) * len(matrix[0])
        )
        for i in range(len(self.matrix)):
            for j in range(len(matrix[0])):
                for k in range(len(matrix)):
                    result.matrix[i][j] += self.matrix[i][k] * matrix[k][j]

        return result

    def scalarMultiply(self, n):
        """
        Multiplies the current matrix by a scalar value.

        Parameters:
        n (float): The scalar value to multiply with.

        Returns:
        Matrix: The resulting matrix after scalar multiplication.
        """
        result = Matrix(len(self.matrix), len(self.matrix[0]))
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                result.matrix[i][j] = self.matrix[i][j] * n

        return result

    def transpose(self):
        """
        Transposes the current matrix.

        Returns:
        Matrix: The transposed matrix.
        """
        r = len(self.matrix)
        c = len(self.matrix[0])
        result = Matrix(c, r)
        for i in range(r):
            for j in range(c):
                result.matrix[j][i] = self.matrix[i][j]
        return result

    def sub_mat(self, matrix, i, j):
        """
        Generates a submatrix by removing the specified row and column.

        Parameters:
        matrix (list): The original matrix.
        i (int): Row index to remove.
        j (int): Column index to remove.

        Returns:
        list: The resulting submatrix.
        """
        return [row[:j] + row[j + 1 :] for row in (matrix[:i] + matrix[i + 1 :])]

    def recursiveDet(self, matrix):
        """
        Computes the determinant of the given matrix recursively.

        Parameters:
        matrix (list): The matrix to compute the determinant for.

        Returns:
        float: The determinant of the matrix.
        """
        r = len(matrix)
        c = len(matrix[0])
        det = 0
        if r == 2 and c == 2:
            return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        # Using cofactor expansion method
        for i in range(c):
            det += (
                ((-1) ** (2 + i))
                * matrix[0][i]
                * self.recursiveDet(self.sub_mat(self.matrix, 0, i))
            )
        return det

    def determinant(self):
        """
        Computes the determinant of the current matrix.

        Returns:
        float: The determinant.

        Raises:
        ValueError: If the matrix is not square.
        """
        r = len(self.matrix)
        c = len(self.matrix[0])
        if r != c:
            raise ValueError("Not a square matrix.")

        return self.recursiveDet(self.matrix)

    def minor(self, matrix):
        """
        Computes the matrix of minors.

        Parameters:
        matrix (list): The matrix to compute minors for.

        Returns:
        Matrix: The matrix of minors.
        """
        r = len(matrix)
        c = len(matrix[0])
        m = Matrix(r, c)
        if r == 2 and c == 2:
            m.matrix = [[matrix[1][1], matrix[0][1]], [matrix[1][0], matrix[0][0]]]
        else:
            m.matrix = [
                [self.recursiveDet(self.sub_mat(matrix, i, j)) for j in range(c)]
                for i in range(r)
            ]
        return m

    def cofactor(self, matrix):
        """
        Computes the matrix of cofactors.

        Parameters:
        matrix (list): The matrix fow which cofactors are to be computed.

        Returns:
        Matrix: The matrix of cofactors.
        """
        r = len(matrix)
        c = len(matrix[0])
        m = self.minor(matrix)
        cof = Matrix(r, c)
        cof.matrix = [
            [((-1) ** (i + j + 2)) * m.matrix[i][j] for j in range(c)] for i in range(r)
        ]
        return cof

    def adjunct(self, matrix):
        """
        Computes the adjunct (adjoint) of the matrix.

        Parameters:
        matrix (list): The matrix fow which the adjunct is to be computed.

        Returns:
        Matrix: The adjunct of the matrix.
        """
        return self.cofactor(matrix).transpose()

    def inverse(self):
        """
        Computes the inverse of the matrix.

        Returns:
        Matrix: The inverse of the matrix.

        Raises:
        ValueError: If the inverse does not exist (determinant is zero).
        """
        if self.determinant() == 0:
            raise ValueError("Inverse does not exist!")
        return self.adjunct(self.matrix).scalarMultiply(1 / self.determinant())

    def equal(self, matrix):
        """
        Checks if the current matrix is equal to another matrix.

        Parameters:
        matrix (Matrix): The matrix to compare with.

        Returns:
        bool: True if matrices are equal, False otherwise.
        """
        a = self.matrix
        b = matrix.matrix

        if len(a) != len(b) or len(a[0]) != len(b[0]):
            return False

        for i in range(len(a)):
            for j in range(len(a[0])):
                if a[i][j] != b[i][j]:
                    return False

        return True
