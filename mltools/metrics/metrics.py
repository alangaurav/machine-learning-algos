class Metrics:
    def __init__(self):
        pass

    def mean_squared_error(self, y, y_preds):
        N = len(y)
        mse = (sum([(y[i] - y_preds[i]) ** 2 for i in range(N)])) / N
        return mse

    def mean_absolute_error(self, y, y_preds):
        N = len(y)
        mae = (sum([(y[i] - y_preds[i]) for i in range(N)])) / N
        return mae

    def confusion_matrix(self, y, y_preds):
        # Assuming binary classification
        N = len(y)
        # TP FP FN TN
        conf_mat = [[0, 0], [0, 0]]
        for i in range(N):
            if y[i] == 'Yes' and y_preds[i] == 'Yes':
                conf_mat[0][0] += 1
            elif y[i] == 'No' and y_preds[i] == 'No':
                conf_mat[1][1] += 1
            elif y[i] == 'No' and y_preds[i] == 'Yes':
                conf_mat[0][1] +=1
            else:
                conf_mat[1][0] += 1

        return conf_mat

    def classification_accuracy(self, y, y_preds):
        conf_mat = self.confusion_matrix(y, y_preds)
        accuracy = (conf_mat[0][0] + conf_mat[1][1])/len(y)

        return accuracy