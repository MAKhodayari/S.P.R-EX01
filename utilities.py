import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#   Global Helpers
def normalize(X):
    NX = pd.DataFrame(columns=X.columns.values)
    for column in NX.columns:
        X_max = X[column].max()
        X_min = X[column].min()
        X_range = X_max - X_min
        if X_range != 0:
            NX[column] = (X[column] - X_min) / X_range
        else:
            NX[column] = X[column] / X_max
    return NX


#   Linear Helpers
def open_linear():
    linear_data = pd.read_csv('./datasets/linear_data.csv', names=['X1', 'y'])
    X = linear_data.iloc[:, :-1]
    y = linear_data.iloc[:, -1]
    return X, y


def find_theta(X, y):
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    return theta


def linear_prediction(X, theta):
    yh = np.dot(X, theta.T)
    return yh


def calc_mse(y, yh):
    m_sample = len(y)
    mse = sum((y - yh) ** 2) / (2 * m_sample)
    return mse


def linear_gradient_descent(X, y, alpha, n_iter):
    m_sample, n_feature = X.shape
    theta = np.random.rand(n_feature)
    iter_list = [i for i in range(n_iter)]
    iter_cost = []
    for i in range(n_iter):
        change = []
        for j in range(n_feature):
            change.append((np.dot((linear_prediction(X, theta) - y), X.iloc[:, j])) / m_sample)
            theta[j] = theta[j] - alpha * change[j]
        iter_cost.append(sum(change))
    iter_cost = iter_cost[10:]
    iter_list = iter_list[10:]
    plt.plot(iter_list, iter_cost, color='red')
    plt.scatter(iter_list[::5], iter_cost[::5], marker='.', color='black')
    plt.suptitle('Gradient Descent')
    plt.title('Cost Per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
    return theta


#   Logistic Helpers
def open_logistic():
    logistic_data = pd.read_csv('./datasets/logistic_data.txt', sep='\t',
                                names=['1', '2', 'X1', '3', '4', '5', 'X2', 'y'])
    logistic_data = logistic_data.drop(['1', '2', '3', '4', '5'], axis=1)
    logistic_data = logistic_data.drop(logistic_data[logistic_data.y == 3].index)
    X = logistic_data.iloc[:, :-1]
    y = logistic_data.iloc[:, -1]
    return X, y


def sigmoid(z):
    ans = 1 / (1 + np.exp(-z))
    return ans


def logistic_prediction(X, theta, norm=False):
    z = np.dot(X, theta.T)
    h = sigmoid(z)
    if not norm:
        return h
    else:
        yh = [0 if label < 0.5 else 1 for label in h]
        return yh


def calc_cross_entropy(X, y, theta):
    m_sample = X.shape[0]
    ones = np.ones(m_sample)
    h = logistic_prediction(X, theta)
    cost = -(np.dot(y.T, np.log10(h)) + np.dot((ones - y).T, np.log10(ones - h))) / m_sample
    return cost


def calc_accuracy(y, yh):
    acc = np.sum(yh == y) / len(y)
    return acc


def logistic_gradient_descent(X, y, alpha, n_iter):
    m_sample, n_feature = X.shape
    theta = np.random.rand(n_feature)
    iter_list = [i for i in range(n_iter)]
    iter_cost = []
    for i in range(n_iter):
        linear_pred = np.dot(X, theta)
        predictions = sigmoid(linear_pred)
        change = []
        for j in range(n_feature):
            change.append((np.dot(X.iloc[:, j], (predictions - y))) / m_sample)
            theta[j] = theta[j] - alpha * change[j]
        iter_cost.append(-sum(change))
    iter_cost = iter_cost[50:]
    iter_list = iter_list[50:]
    plt.plot(iter_list, iter_cost, color='red')
    plt.scatter(iter_list[::500], iter_cost[::500], marker='.', color='black')
    plt.suptitle('Gradient Descent')
    plt.title('Cost Per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
    return theta
