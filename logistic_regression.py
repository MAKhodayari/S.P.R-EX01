import numpy as np
import pandas as pd
import utilities as utl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    #   Opening & Preparing Data
    X, y = utl.open_logistic()

    X = utl.normalize(X)
    X.insert(0, 'X0', 1)

    y = pd.Series([0 if label == 1 else 1 for label in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #   Gradient Descent Solution
    gd_theta = utl.logistic_gradient_descent(X_train, y_train, 0.5, 10000)

    gd_ce_train = utl.calc_cross_entropy(X_train, y_train, gd_theta)
    gd_ce_test = utl.calc_cross_entropy(X_test, y_test, gd_theta)

    gd_yh_train = utl.logistic_prediction(X_train, gd_theta, True)
    gd_yh_test = utl.logistic_prediction(X_test, gd_theta, True)

    gd_train_acc = utl.calc_accuracy(y_train, gd_yh_train)
    gd_test_acc = utl.calc_accuracy(y_test, gd_yh_test)

    print(f'Theta: {gd_theta} | Train Cross Entropy: {gd_ce_train} | Test Cross Entropy: {gd_ce_test}')
    print(f'Train Accuracy {gd_train_acc} | Test Accuracy {gd_test_acc}')

    #   Figures & Plots
    train_decision_boundary = - (gd_theta[0] + np.dot(gd_theta[1], X_train['X1'])) / gd_theta[2]
    test_decision_boundary = - (gd_theta[0] + np.dot(gd_theta[1], X_test['X1'])) / gd_theta[2]

    fig, ax = plt.subplots(1, 2, sharex='all', sharey='all')

    fig.suptitle('Logistic Regression')

    ax[0].scatter(X_train['X1'], X_train['X2'], c=y_train)
    ax[0].plot(X_train['X1'], train_decision_boundary)
    ax[0].set_title('Train Set')
    ax[0].set(xlabel='X[X1]', ylabel='X[X2]')

    ax[1].scatter(X_test['X1'], X_test['X2'], c=y_test)
    ax[1].plot(X_test['X1'], test_decision_boundary)
    ax[1].set_title('Test Set')
    ax[1].set(xlabel='X[X1]', ylabel='X[X2]')

    fig.tight_layout()

    plt.show()
