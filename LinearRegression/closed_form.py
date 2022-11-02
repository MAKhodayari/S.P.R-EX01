import linear_regression as lin_reg
import utilities as utl


cf_theta = utl.find_theta(lin_reg.X_train, lin_reg.y_train)

cf_yh_train = utl.predict(lin_reg.X_train, cf_theta)
cf_yh_test = utl.predict(lin_reg.X_test, cf_theta)

cf_mse_train = utl.calc_mse(lin_reg.y_train, cf_yh_train)
cf_mse_test = utl.calc_mse(lin_reg.y_test, cf_yh_test)

print(f'Theta: {cf_theta} | Train MSE: {cf_mse_train} | Test MSE: {cf_mse_test}')
