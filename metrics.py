class Metrics:
    def __init__(self):
        pass

    def mean_squared_error(y, y_preds):
        N = len(y)
        mse = (sum([(y_i - y_j)**2 for y_i in y for y_j in y_preds]))/N
        return mse