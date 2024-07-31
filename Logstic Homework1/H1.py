import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

def load_breast_cancer_scaled(add_intercept):
    data = load_breast_cancer()
    X, t = data.data, data.target
    n_classes = data.target.size
    t = (t == 1).astype(int)  # 'malignant' is labeled as 1 in the dataset

    X = MinMaxScaler().fit_transform(X)  # Scaling the data

    if add_intercept:  # Add intercept after scaling
        X = np.hstack([np.ones((X.shape[0], 1)), X])

    X_train, X_test, y_train, y_test = train_test_split(X, t, test_size=0.3,
                                                        shuffle=True, stratify=t,
                                                        random_state=0)

    return X_train, X_test, y_train, y_test

def sigmoid_fun(z):
    return 1 / (1 + np.exp(-z))

def cost_f(X, t, weights):
    examples = X.shape[0]
    pred = sigmoid_fun(np.dot(X, weights))
    positive = -t * np.log(pred)
    negative = -(1 - t) * np.log(1 - pred)
    cost = 1 / examples * np.sum(positive + negative)
    return cost

def f_dervative(X, t, weights):
    examples = X.shape[0]
    pred = sigmoid_fun(np.dot(X, weights))
    error = pred - t
    gradient = X.T @ error / examples
    return gradient

def gradient_descent_logstic_regression(X, t, step_size=0.1, precision=0.0001, max_iter=7000):  # no changes. Different params
    examples, features = X.shape
    iter = 0
    cur_weights = np.random.rand(features)  # random starting point
    last_weights = cur_weights + 100 * precision  # something different

    print(f'Initial Random Cost: {cost_f(X, t, cur_weights)}')

    while norm(cur_weights - last_weights) > precision and iter < max_iter:
        last_weights = cur_weights.copy()  # must copy
        gradient = f_dervative(X, t, cur_weights)
        cur_weights -= gradient * step_size
        iter += 1

    print(f'Total Iterations {iter}')
    print(f'Optimal Cost: {cost_f(X, t, cur_weights)}')
    return cur_weights

def accuracy(X, t, weights, threshold=0.5):
    m = X.shape[0]
    prop = sigmoid_fun(np.dot(X, weights))
    labels = (prop >= threshold).astype(int)
    correct = np.sum((t == labels))
    return correct / m * 100.0

if __name__ == '__main__':
    add_intercept = False  # Try with and without for our model!
    X_train, X_test, y_train, y_test = load_breast_cancer_scaled(add_intercept)
    optimal_weights = gradient_descent_logstic_regression(X_train, y_train)
    test_accuracy = accuracy(X_test, y_test, optimal_weights)
    print(f'Test Accuracy: {test_accuracy:.2f}%')
