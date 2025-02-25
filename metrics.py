class Metrics:
    def __init__(self):
        pass

    def mean_squared_error(self, y, y_preds):
        N = len(y)
        mse = (sum([(y[i] - y_preds[i]) ** 2 for i in range(N)])) / N
        return mse
