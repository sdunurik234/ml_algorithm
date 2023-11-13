import numpy as np


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))


class LogLoss():
    def __call__(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # dLoss/dw = 1/m*summa(o(w*x)-y_pred)*x
    # d - differential
    # m - size of x or y
    # o(z) = sigmoid function
    # x - factor(data)
    # y_pred - predictions

    # gradient of the function in matrix form
    # X^(T)*(o(X*W)-Y)
    # X^(T) - transpose of X
    # o(z) - sigmoid(z) activation
    # W - weight
    # X - features(data)
    # Y - true y(i)


class MyLogisticRegression():
    def __init__(self, learning_rate: int = 0.001, n_iter: int = 2000) -> None:
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w = None
        self.b = None

        self.activation = Sigmoid()

    def fit(self, X, y):
        num_s, num_f = X.shape
        self.w = np.random.rand(num_f)
        self.b = 0

        for i in range(self.n_iter):
            y_hat = np.dot(X, self.w.T) + self.b
            y_pred = self.activation(y_hat)

            dw = (1 / num_s) * np.dot(X.T, y_pred - y)
            db = (1 / num_s) * np.sum(y_pred - y)

            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

        return self

    def predict(self, X):
        return self.activation(np.dot(X, self.w.T) + self.b)

X = np.array([[16],[29],[3],[4],[9],[13]])
y = np.array([1, 0, 0, 1, 1, 0])

log = MyLogisticRegression()
log.fit(X, y)

log.predict([[89]])

from sklearn.linear_model import LogisticRegression

sklog = LogisticRegression()
sklog.fit(X, y)

sklog.predict([[89]])