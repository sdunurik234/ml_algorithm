import numpy as np


class MSE():
    def __call__(self, y_true, X, w, f):
        return (1 / f) * np.sum((y_true - np.dot(X, w)) ** 2)

    def gradient(self, y_true, X, w):
        return (2 / num) * np.dot((y_true - np.dot(X, w)), (-X))

class MAE():
    def __call__(self, y_true, X, w, b):
        return (1 / num_s) * np.sum((np.dot(X, self.w) + self.b) - y)

    def gradient(self, y_true, X, w, b):
        return (1 / num_s) * np.dot(X.T, (np.dot(X, self.w) + self.b) - y)

class LinearRegressionMSE:
    def __init__(self, learning_rate: int = 0.001, n_iter: int = 2000) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        num_s, num_f = X.shape
        self.w = np.random.rand(num_f)
        self.b = 0

        for i in range(self.n_iter):
            y_pred = np.dot(X, self.w)

            dw = (2 / num_s) * np.dot((y - y_pred), (-X))
            db = (1 / num_s) * np.sum((y - y_pred) ** 2)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b


class LinearRegressionMAE:
    def __init__(self, learning_rate: int = 0.001, n_iter: int = 2000) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        num_s, num_f = X.shape
        self.w = np.random.rand(num_f)
        self.b = 0

        for i in range(self.n_iter):
            y_pred = np.dot(X, self.w) + self.b

            dw = (1 / num_s) * np.dot(X.T, y_pred - y)
            db = (1 / num_s) * np.sum(y_pred - y)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

        return self

    def predict(self, X):
        return np.dot(X, self.w) + self.b

X = np.array([[1],[2],[3],[4],[6],[8],[10]])
y = np.array([2,4,6,8,12,16,20])

linearmse = LinearRegressionMSE()
linearmae = LinearRegressionMAE()

linearmae.fit(X, y)
linearmse.fit(X, y)

print(f"LinearRegression MSE: {round(linearmse.predict([20]), 2)}, Weight: {round(linearmse.w[0], 2)}, Intercept: {round(linearmse.b, 2)}")
print(f"LinearRegression MAE: {round(linearmae.predict([20]), 2)}, Weight: {round(linearmae.w[0], 2)}, Intercept: {round(linearmae.b, 2)}")