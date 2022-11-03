import utilities as utl
import logistic_regression as log_reg


grad_theta = utl.logistic_gradient_ascent(log_reg.X_train, log_reg.y_train, 0.9, 1000, 10 ** -10)

grad_yh_train = utl.logistic_prediction(log_reg.X_train, grad_theta)
grad_yh_test = utl.logistic_prediction(log_reg.X_test, grad_theta)

grad_mse_train = utl.calc_mse(log_reg.y_train, grad_yh_train)
grad_mse_test = utl.calc_mse(log_reg.y_test, grad_yh_test)

print(f'Theta: {grad_theta} | Train MSE: {grad_mse_train} | Test MSE: {grad_mse_test}')
