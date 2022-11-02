import numpy as np
import pandas as pd


def open_linear():
    linear_data = pd.read_csv('../datasets/linear_data.csv', names=['X1', 'y'])
    X = linear_data.iloc[:, :-1]
    y = linear_data.iloc[:, -1]
    return X, y


def normalize(X):
    NX = pd.DataFrame(columns=X.columns.values)
    n_feature = len(X.iloc[0])
    for i in range(1, n_feature + 1):
        X_max = X['X' + str(i)].max()
        X_min = X['X' + str(i)].min()
        X_range = X_max - X_min
        if X_range != 0:
            NX['X' + str(i)] = (X['X' + str(i)] - X_min) / X_range
        else:
            NX['X' + str(i)] = X['X' + str(i)] / X_max
    return NX


def find_theta(X, y):
    theta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    return theta


def predict(X, theta):
    yh = []
    m_sample = len(X)
    for i in range(m_sample):
        yh.append(np.dot(theta.T, X.iloc[i]))
    return yh


def calc_mse(y, yh):
    m_sample = len(y)
    mse = sum((y - yh) ** 2) / (2 * m_sample)
    return mse


def gradient(X, y, alpha, n_iter, eps):
    m_sample = len(X)
    n_feature = len(X.iloc[0])
    theta = np.random.rand(n_feature)
    for i in range(n_iter):
        for j in range(n_feature):
            sample_cost = []
            for k in range(m_sample):
                sample_cost.append((np.dot(theta.T, X.iloc[k]) - y.iloc[k]) * X.iloc[k, j])
            cost = sum(sample_cost) / m_sample
            if abs(cost) > eps:
                theta[j] = theta[j] + alpha * cost
            else:
                return theta
    return theta
