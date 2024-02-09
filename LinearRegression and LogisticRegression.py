import numpy as np
class LinearRegression:
    def __init__(self, learning_rate = 0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weight = None
        self.bais = None

    def fit(self,X, y):
        n_sample, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bais = 0

        for i in range(self.n_iterations):
            y_predicted = np.dot(X,self.weight)+self.bais
            error = y_predicted - y

            dw = (1/n_sample)*np.dot(X.T, (error))
            db = (1/n_sample)*np.sum(error)

            self.weight -= self.learning_rate*dw
            self.bais -= self.learning_rate*db

    def predict(self,X):
        return np.dot(X, self.weight) + self.bais

X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([2, 3.5, 4.5])

model = LinearRegression()
model.fit(X, y)

new_data = np.array([[4, 5]])
print(model.predict(new_data))


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iteration=1000):
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        n_sample, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        # Feature scaling
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X_scaled = (X - self.X_mean) / self.X_std

        for i in range(self.n_iteration):
            z = np.dot(X_scaled, self.weight) + self.bias
            y_predicted = 1 / (1 + np.exp(-z))
            error = y_predicted - y
            dw = (1 / n_sample) * np.dot(X_scaled.T, error)
            db = (1 / n_sample) * np.sum(error)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Feature scaling
        X_scaled = (X - self.X_mean) / self.X_std

        z = np.dot(X_scaled, self.weight) + self.bias
        return 1 / (1 + np.exp(-z))


X = np.array([[25, 50000], [35, 75000], [50, 100000], [40, 80000]])
y = np.array([0, 0, 1, 1])
model = LogisticRegression()
model.fit(X, y)
new_data = np.array([[55, 120000]])
print(model.predict(new_data))