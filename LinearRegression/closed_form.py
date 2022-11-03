import utilities as utl
import linear_regression as lin_reg


theta = utl.find_theta(lin_reg.X_train, lin_reg.y_train)

yh_train = utl.linear_prediction(lin_reg.X_train, theta)
yh_test = utl.linear_prediction(lin_reg.X_test, theta)

mse_train = utl.calc_mse(lin_reg.y_train, yh_train)
mse_test = utl.calc_mse(lin_reg.y_test, yh_test)

print(f'Closed Form Info:\nTheta = {theta} | Train MSE = {mse_train} | Test MSE = {mse_test}')
