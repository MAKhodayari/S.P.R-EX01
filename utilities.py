import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def open_linear():
    linear_data = pd.read_csv('../datasets/linear_data.csv', names=['X1', 'y'])
    X = linear_data.iloc[:, :-1]
    y = linear_data.iloc[:, -1]
    return X, y


def open_logistic():
    logistic_data = pd.read_csv('../datasets/logistic_data.txt', sep='\t',
                                names=['1', '2', 'X1', '3', '4', '5', 'X2', 'y'])
    logistic_data = logistic_data.drop(['1', '2', '3', '4', '5'], axis=1)
    logistic_data = logistic_data.drop(logistic_data[logistic_data.y == 3].index)
    X = logistic_data.iloc[:, :-1]
    y = logistic_data.iloc[:, -1]
    return X, y


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


def sigmoid(z):
    ans = 1 / (1 + np.exp(-z))
    return ans


def logistic_prediction(X, theta):
    z = np.dot(X, theta.T)
    yh = sigmoid(z)
    return yh


def logistic_likelihood(y, yh):
    m_sample = len(y)
    sample_likelihood = []
    for i in range(m_sample):
        sample_likelihood.append(y.iloc[i] * np.log10(yh[i]) + (1 - y.iloc[i]) * np.log10(1 - yh[i]))
    likelihood = sum(sample_likelihood) / m_sample
    return likelihood


def linear_gradient_descent(X, y, alpha, n_iter):
    m_sample = len(X)
    n_feature = len(X.iloc[0])
    theta = np.random.rand(n_feature)
    iter_cost = []
    iter_num = [i for i in range(n_iter)]
    for i in range(n_iter):
        cost = 0
        for j in range(n_feature):
            sample_cost = np.dot((linear_prediction(X, theta) - y), X.iloc[:, j])
            cost += np.sum(sample_cost) / m_sample
            theta[j] = theta[j] - alpha * cost
        iter_cost.append(cost)
    iter_num = iter_num[5:]
    iter_cost = iter_cost[5:]
    plt.scatter(iter_num[::5], iter_cost[::5], marker='.', color='black')
    plt.plot(iter_num, iter_cost, color='red')
    plt.title('Cost Per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
    return theta


def logistic_gradient_ascent(X, y, alpha, n_iter, eps):
    m_sample = len(X)
    n_feature = len(X.iloc[0])
    theta = np.random.rand(n_feature)
    for i in range(n_iter):
        for j in range(n_feature):
            sample_cost = []
            for k in range(m_sample):
                sample_cost.append(np.dot((y.iloc[k] - sigmoid(np.dot(theta.T, X.ilox[k]))), X.iloc[k]))
            cost = sum(sample_cost)
            if abs(cost) > eps:
                theta[j] = theta[j] + alpha * cost
            else:
                return theta
    return theta
