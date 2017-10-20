from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import pandas as pd


#load data
def load_dataset(csv_path):
    return pd.read_csv(csv_path, delimiter=';')


#distance loss
def dist_loss(y, y_):
    y, y_ = K.flatten(y), K.flatten(y_)
    return K.sqrt(K.sum(K.square(y - y_)))


def mean_loss(y, y_):
    y, y_ = K.flatten(y), K.flatten(y_)
    return K.mean(K.square(y - y_))


def dist_error(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) ** 2))


def mean_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


#neural net model
def reg(input_shape, loss_func, output_shape, act_function='tanh'):
    model = Sequential()

    model.add(Dense(10, input_shape=(input_shape,)))
    model.add(Activation(act_function))
    # model.add(Dropout(0.8))

    # model.add(Dense(5, input_shape=(input_shape,)))
    # model.add(Activation(act_function))
    # model.add(Dropout(0.8))

    # model.add(Dense(20, input_shape=(input_shape,)))
    # model.add(Activation(act_function))
    # model.add(Dropout(0.8))

    model.add(Dense(output_shape))
    model.add(Activation(act_function))

    ada = optimizers.Adagrad(lr=0.01)

    model.compile(optimizer=ada, loss=loss_func)
    return model


#standard scaler and transformation
def sctrans(X, trans):

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    X_ = trans.fit_transform(X_scaled)

    return X_scaled, X_


#cv
def cv_function(mod, cv, X, y, metric_func):
    
    #variables
    metric_kfold = list()
    cv_predictions = np.zeros_like(y)

    for fold, (train_index, test_index) in enumerate(cv.split(X, y)):
        print(fold)
        print(train_index)
        mod.fit(X[train_index], y[train_index])

        #metrics
        predictions = mod.predict(X[test_index])
        metric_kfold.append(metric_func(y[test_index], predictions))
        cv_predictions[test_index] = predictions

    return metric_kfold, cv_predictions



if __name__ == '__main__':

    # dataset
    dataset = load_dataset('datasets/winequality-white.csv')
    X, y = dataset.drop('quality', axis=1).values , dataset['quality'].values

    # #scaler + transformation
    # #choose one transformation
    n_comp = 2
    pca = PCA(n_components=n_comp)
    md = MDS(n_components=n_comp, random_state=0, n_jobs=-1, verbose=10)

    X_scaled, X_ = sctrans(X, md)

    #neural net
    nn = lambda: reg(X_scaled.shape[1], output_shape=n_comp, loss_func=dist_loss, act_function='linear')
    model = KerasRegressor(nn, epochs=50, batch_size=50)    

    #fold dataset
    kf = KFold(n_splits=3, shuffle=True)
    error_kfold, cv_preds = cv_function(model, kf, X_scaled, X_, dist_error)

    #plot
    plt.scatter(X_[:,0], X_[:,1])
    plt.scatter(cv_preds[:,0], cv_preds[:,1], alpha=0.5)
    plt.show()
