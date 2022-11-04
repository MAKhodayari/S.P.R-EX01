import utilities as utl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    #   Opening & Preparing Data
    X, y = utl.open_linear()

    X = utl.normalize(X)
    X.insert(0, 'X0', 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #   Closed Form Solution
    cf_theta = utl.find_theta(X_train, y_train)

    cf_yh_train = utl.linear_prediction(X_train, cf_theta)
    cf_yh_test = utl.linear_prediction(X_test, cf_theta)

    cf_mse_train = utl.calc_mse(y_train, cf_yh_train)
    cf_mse_test = utl.calc_mse(y_test, cf_yh_test)

    print(f'Closed Form Info:\nTheta = {cf_theta} | Train MSE = {cf_mse_train} | Test MSE = {cf_mse_test}')

    #   Gradient Descent Solution
    gd_theta = utl.linear_gradient_descent(X_train, y_train, 0.5, 200)

    gd_yh_train = utl.linear_prediction(X_train, gd_theta)
    gd_yh_test = utl.linear_prediction(X_test, gd_theta)

    gd_mse_train = utl.calc_mse(y_train, gd_yh_train)
    gd_mse_test = utl.calc_mse(y_test, gd_yh_test)

    print(f'Gradient Descent Info:\nTheta = {gd_theta} | Train MSE = {gd_mse_train} | Test MSE = {gd_mse_test}')

    #   Figures & Plots
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')

    fig.suptitle('Linear Regression: Closed Form VS Gradient Descent')

    axs[0, 0].scatter(X_train['X1'], y_train, color='green', marker='.')
    axs[0, 0].plot(X_train['X1'], cf_yh_train, color='black', linestyle='dashed')
    axs[0, 0].set_title('Closed Form Train Set')
    axs[0, 0].set(xlabel='X[X1]', ylabel='y')

    axs[1, 0].scatter(X_test['X1'], y_test, color='green', marker='.')
    axs[1, 0].plot(X_test['X1'], cf_yh_test, color='black', linestyle='dashed')
    axs[1, 0].set_title('Closed Form Test Set')
    axs[1, 0].set(xlabel='X[X1]', ylabel='y')

    axs[0, 1].scatter(X_train['X1'], y_train, color='orange', marker='.')
    axs[0, 1].plot(X_train['X1'], gd_yh_train, color='blue', linestyle='dashed')
    axs[0, 1].set_title('Gradient Descent Train Set')
    axs[0, 1].set(xlabel='X[X1]', ylabel='y')

    axs[1, 1].scatter(X_test['X1'], y_test, color='orange', marker='.')
    axs[1, 1].plot(X_test['X1'], gd_yh_test, color='blue', linestyle='dashed')
    axs[1, 1].set_title('Gradient Descent Test Set')
    axs[1, 1].set(xlabel='X[X1]', ylabel='y')

    fig.tight_layout()

    plt.show()
