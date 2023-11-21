import numpy as np


class KNNRegressor:
    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            d = np.linalg.norm(self.X - x, axis=1)
            index = np.argsort(d)[:self.k]
            neighbors_y = self.y[index]
            prediction = np.mean(neighbors_y)
            predictions.append(prediction)
        return np.array(predictions)

X = np.array([[1],[2],[3],[4],[6],[8],[10]])
y = np.array([2,4,6,8,12,16,20])
knn_reg = KNNRegressor()
knn_reg.fit(X, y)
knn_reg.predict([[6.5]])