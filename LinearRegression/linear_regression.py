import utilities as ut
from sklearn.model_selection import train_test_split


X, y = ut.open_linear()

NX = ut.normalize(X)
NX.insert(0, 'X0', 1)

X_train, X_test, y_train, y_test = train_test_split(NX, y, test_size=0.3)
