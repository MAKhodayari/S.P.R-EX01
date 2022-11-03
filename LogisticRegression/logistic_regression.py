import utilities as utl
from sklearn.model_selection import train_test_split


X, y = utl.open_logistic()

X = utl.normalize(X)
X.insert(0, 'X0', 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
