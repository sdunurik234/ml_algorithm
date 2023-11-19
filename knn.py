import numpy as np

class KNN:
    def __init__(self, k: int = 3) -> None:
        self.k = k

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        predictions = []
        for x in X:
            d = [np.linalg.norm(x - i) for i in self.X]  # euclidean distance formula
            k_index = np.argsort(d)[:self.k]
            k_label = self.y[k_index]
            mc = np.bincount(k_label).argmax()
            predictions.append(mc)
        return np.array(predictions)

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([0, 0, 1, 1])

knn = KNN()
knn.fit(X, y)
knn.predict([[3, 4]])
