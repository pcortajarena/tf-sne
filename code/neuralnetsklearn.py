from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Activation
from keras import optimizers
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# cual es la diferencia con StratifiedKFold
# StratifiedKFold es para clasificacion, ya lo veremos


def reg():
    model = Sequential()

    model.add(Dense(1, input_shape=(13,)))
    model.add(Activation('linear'))

    ada = optimizers.Adagrad(lr=1)

    model.compile(optimizer=ada, loss='mean_squared_error')
    return model


def cv_function(mod, cv, X, y, metric_func):

    # variables
    metric_kfold = list()
    cv_predictions = np.zeros_like(y)

    for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
        print(fold)
        mod.fit(X[train_index], y[train_index])

        # metrics
        predictions = mod.predict(X[test_index])
        metric_kfold.append(metric_func(y[test_index], predictions))
        cv_predictions[test_index] = predictions

    return metric_kfold, cv_predictions


if __name__ == '__main__':

    # dataset
    boston = load_boston()

    # linear regresion pipeline
    lr = KerasRegressor(reg, epochs=20, batch_size=100)
    model = make_pipeline(StandardScaler(), lr)
    X, y = boston['data'], boston['target']

    # fold dataset
    kf = KFold(n_splits=5, shuffle=True)

    mse_kfold, cv_preds = cv_function(model, kf, X, y, mean_squared_error)
