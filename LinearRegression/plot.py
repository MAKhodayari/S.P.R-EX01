import linear_regression as lin_reg
import gradient_descent as gd
import closed_form as cf
import matplotlib.pyplot as plt


fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')

fig.suptitle('Linear Regression: Closed Form VS Gradient Descent')

axs[0, 0].scatter(lin_reg.X_train['X1'], lin_reg.y_train, color='green', marker='.')
axs[0, 0].plot(lin_reg.X_train['X1'], cf.yh_train, color='black', linestyle='dashed')
axs[0, 0].set(xlabel='X[X1]', ylabel='y')
axs[0, 0].set_title('Closed Form Train Set')

axs[1, 0].scatter(lin_reg.X_test['X1'], lin_reg.y_test, color='green', marker='.')
axs[1, 0].plot(lin_reg.X_test['X1'], cf.yh_test, color='black', linestyle='dashed')
axs[1, 0].set(xlabel='X[X1]', ylabel='y')
axs[1, 0].set_title('Closed Form Test Set')


axs[0, 1].scatter(lin_reg.X_train['X1'], lin_reg.y_train, color='orange', marker='.')
axs[0, 1].plot(lin_reg.X_train['X1'], gd.yh_train, color='blue', linestyle='dashed')
axs[0, 1].set(xlabel='X[X1]', ylabel='y')
axs[0, 1].set_title('Gradient Descent Train Set')

axs[1, 1].scatter(lin_reg.X_test['X1'], lin_reg.y_test, color='orange', marker='.')
axs[1, 1].plot(lin_reg.X_test['X1'], gd.yh_test, color='blue', linestyle='dashed')
axs[1, 1].set(xlabel='X[X1]', ylabel='y')
axs[1, 1].set_title('Gradient Descent Test Set')

fig.tight_layout()

plt.show()
