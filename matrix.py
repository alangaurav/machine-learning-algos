# Matrix
class Matrix:
    def __init__(self, n=None, m=None, data=None):
        if n == None or m == None:
            self.matrix = None
            return
        elif data is not None:
            matrix = [[data[i*(m-1) + j + i] for j in range(m)] for i in range(n)]
        else:
            matrix = [[0 for _ in range(m)] for _ in range(n)]
        self.matrix = matrix

    def multiply(self, matrix):
        matrix = matrix.matrix
        if len(self.matrix[0]) != len(matrix):
            raise ValueError("Incompatible matrices")

        result = Matrix(len(self.matrix), len(matrix[0]), [0] * len(self.matrix) * len(matrix[0]))
        for i in range(len(self.matrix)):
            for j in range(len(matrix[0])):
                for k in range(len(matrix)):
                    result.matrix[i][j] += self.matrix[i][k] * matrix[k][j]

        return result

    def scalarMultiply(self, n):
        result = Matrix(len(self.matrix), len(self.matrix[0]))
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[0])):
                result.matrix[i][j] = self.matrix[i][j] * n

        return result

    def transpose(self):
        r = len(self.matrix)
        c = len(self.matrix[0])
        result = Matrix(c, r)
        for i in range(r):
            for j in range(c):
                result.matrix[j][i] = self.matrix[i][j]
        return result

    def sub_mat(self, matrix, i, j):
        return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

    def recursiveDet(self, matrix):
        r = len(matrix)
        c = len(matrix[0])
        det = 0
        if r == 2 and c == 2:
            return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        # Using cofactor expansion method
        for i in range(c):
            det +=  ((-1) ** (2 + i)) * matrix[0][i] * self.recursiveDet(self.sub_mat(self.matrix, 0, i))
        return det

    def determinant(self):
        r = len(self.matrix)
        c = len(self.matrix[0])
        if r != c:
            raise ValueError("Not a square matrix.")

        return self.recursiveDet(self.matrix)

    def minor(self, matrix):
        r = len(matrix)
        c = len(matrix[0])
        if r == 2 and c == 2:
            return matrix
        m = Matrix(r, c)
        m.matrix = [[self.recursiveDet(self.sub_mat(matrix, i, j)) for j in range(c)] for i in range(r)]
        return m

    def cofactor(self, matrix):
        r = len(matrix)
        c = len(matrix[0])
        m = self.minor(matrix)
        cof = Matrix(r, c)
        cof.matrix = [[((-1) ** (i + j + 2)) * m.matrix[i][j] for j in range(c)] for i in range(r)]
        return cof     

    def adjunct(self, matrix):
        return self.cofactor(matrix).transpose()

    def inverse(self):
        if self.determinant() == 0:
            raise ValueError("Inverse does not exist!")
        return self.adjunct(self.matrix).scalarMultiply(1/self.determinant())
