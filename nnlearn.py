from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import pandas as pd
import numba

#load data
def load_dataset(csv_path):
    return pd.read_csv(csv_path, delimiter=';')


#distance loss
def dist_loss(y, y_):
    # y, y_ = K.flatten(y), K.flatten(y_)
    return K.mean(K.sqrt(K.sum(K.square(y - y_), axis=1)))


def dist_error(y_true, y_pred):
    return np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)))


def mean_loss(y, y_):
    y, y_ = K.flatten(y), K.flatten(y_)
    return K.mean(K.square(y - y_))


def mean_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


#standard scaler and transformation
def sctrans(X, trans):

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    X_ = trans.fit_transform(X_scaled)

    return X_scaled, X_


#neural net model
def reg(shape, loss_func, layers, dropout, output_shape, lr, act_function='tanh'):
    
    model = Sequential()
    for i, (layer, drop) in enumerate(zip(layers, dropout)):

        if i == 0:
            model.add(Dense(layer, input_shape=(shape,), activation=act_function))
        else:
            model.add(Dense(layer, activation=act_function))

        model.add(Dropout(drop))

    model.add(Dense(output_shape, activation='linear'))
    ada = optimizers.Adagrad(lr=lr)

    model.compile(optimizer=ada, loss=loss_func)
    return model


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


def plot_similarity(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true[:,0], y_true[:,1])
    plt.scatter(y_pred[:,0], y_pred[:,1], alpha=0.5)
    plt.savefig('similarity.png')


def calculate_corrcoef(y_true, y_pred):
    x = (np.corrcoef(y_true[:,0], y_pred[:,0]))
    y = print(np.corrcoef(y_true[:,1], y_pred[:,1]))    
    return x, y


def pairwise_distances_error(y_true, y_pred):
    pw_y_true = pairwise_distances(y_true)
    pw_y_pred = pairwise_distances(y_pred)
    return pw_y_true, pw_y_pred, (pw_y_true - pw_y_pred) / pw_y_true * 100


if __name__ == '__main__':

    # dataset
    dataset = load_dataset('datasets/winequality-white.csv')
    X, y = dataset.drop('quality', axis=1).values , dataset['quality'].values

    # #scaler + transformation
    # #choose one transformation
    n_comp = 2
    pca = PCA(n_components=n_comp)
    md = MDS(n_components=n_comp, random_state=0, n_jobs=-1, verbose=10)
    tsne = TSNE(n_components=n_comp, verbose=10)
    pipeline = make_pipeline(PCA(n_components=5), tsne)

    X_scaled, X_ = sctrans(X, pipeline)

    #neural net
    nn = lambda: reg(
        X_scaled.shape[1], loss_func=mean_loss, 
        layers=[1000], dropout=[0], 
        output_shape=n_comp, lr=0.3, act_function='sigmoid')
    model = KerasRegressor(nn, epochs=200, batch_size=10)    

    #fold dataset
    kf = KFold(n_splits=3, shuffle=True)
    error_kfold, cv_preds = cv_function(model, kf, X_scaled, X_, mean_error)
    print(error_kfold)

    #plot
    plot_similarity(X_, cv_preds)

    #corcoef
    print(calculate_corrcoef(X_, cv_preds))

    #distances
    # print(pairwise_distances_error(X_, cv_preds))


