import linear_regression as lin_reg
import utilities as utl


grad_theta = utl.gradient(lin_reg.X_train, lin_reg.y_train, -0.9, 200, 10 ** -5)

grad_yh_train = utl.predict(lin_reg.X_train, grad_theta)
grad_yh_test = utl.predict(lin_reg.X_test, grad_theta)

grad_mse_train = utl.calc_mse(lin_reg.y_train, grad_yh_train)
grad_mse_test = utl.calc_mse(lin_reg.y_test, grad_yh_test)

print(f'Theta: {grad_theta} | Train MSE: {grad_mse_train} | Test MSE: {grad_mse_test}')
