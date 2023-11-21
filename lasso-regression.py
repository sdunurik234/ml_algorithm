import numpy as np

class LassoRegression:
    def __init__(self, alpha, learning_rate: int = 0.001, n_iter: int = 2000) -> None:
        self.lr = learning_rate
        self.n_iter = n_iter
        self.w = None
        self.w = None
        self.alpha = alpha

    def fit(self, X, y):
        num_s, num_f = X.shape
        self.w = np.random.rand(num_f)
        self.b = 0

        for i in range(self.n_iter):
            y_pred = np.dot(X, self.w) + self.b

            dw = (1 / num_s) * np.dot(X.T, y_pred - y) + self.alpha * np.sign(self.w)
            db = (1 / num_s) * np.sum(y_pred - y)

            self.w = self.w - self.lr * dw
            self.b = self.b - self.lr * db

        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b

X = np.array([[1],[2],[3],[4],[6],[8],[10]])
y = np.array([2,4,6,8,12,16,20])
lasso = LassoRegression(alpha=0.1)
lasso.fit(X, y)
lasso.predict([[7]])