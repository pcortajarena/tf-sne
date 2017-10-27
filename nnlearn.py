from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.datasets import load_boston
from scipy.io.arff import loadarff
from sklearn.model_selection import KFold
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import pandas as pd
import numba


#load data
def load_wines(path):
    dataset = pd.read_csv(path, sep=';')
    X, y = dataset.drop('quality', axis=1).values , dataset['quality'].values
    return X, y


def load_page(path):
    dataset = pd.read_csv(path)
    X, y = dataset.iloc[:,:-1].values, dataset.iloc[:,-1].values
    return X, y


def load_bankruptcy(path):
    dataset = pd.read_csv(path, delimiter='\t')
    X, y = dataset.drop('class', axis=1).values , dataset['class'].values
    return X, y


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
    corrcoefs = [np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1] for i in range(y_true.shape[1])]
    return corrcoefs


def pairwise_distances_error(y_true, y_pred):
    pw_y_true = pairwise_distances(y_true)
    pw_y_pred = pairwise_distances(y_pred)
    return pw_y_true, pw_y_pred, (pw_y_true - pw_y_pred) / pw_y_true * 100


def devidedisin(x, split=None):
    if not split:
        split = int(x.shape[0]/2)
    return np.isin(x[split:], x[0:split])


def knn_percentage_preserved(y_true, y_pred, n_true=100, n_pred=None):

    # If no n_pred is defined, use the same n
    if not n_pred:
        n_pred = n_true

    pw_y_true = pairwise_distances(y_true)
    pw_y_pred = pairwise_distances(y_pred)

    temp_y_true = np.argpartition(pw_y_true, n_true, axis=0)[:n_true]
    temp_y_pred = np.argpartition(pw_y_pred, n_pred, axis=0)[:n_pred]

    conc =  np.concatenate((temp_y_pred, temp_y_true), axis=0)
    return np.apply_along_axis(devidedisin, 0, conc, split=n_pred).mean(axis=0)


if __name__ == '__main__':

    # dataset
    # X, y = load_wines('datasets/winequality-white.csv')
    X, y = load_bankruptcy('datasets/bankruptcy.data')
    # X, y = load_page('datasets/datablocks.csv')


    # #scaler + transformation
    # #choose one transformation and create and unsupervised pipeline
    n_comp = 10
    pca = PCA(n_components=n_comp)
    md = MDS(n_components=n_comp, random_state=0, n_jobs=-1, verbose=10, n_init=4)
    tsne = TSNE(n_components=n_comp, verbose=10, method='exact')

    trans_pipeline = make_pipeline(
        Imputer(),
        StandardScaler()
        )

    unsupervised_pipeline = make_pipeline(
        md
    )

    X_trans = trans_pipeline.fit_transform(X)

    X_ = unsupervised_pipeline.fit_transform(X_trans)

    #neural net;
    nn = lambda: reg(
        X.shape[1], loss_func=dist_loss,
        layers=[1000], dropout=[0],
        output_shape=n_comp, lr=0.1, act_function='sigmoid')
    model = KerasRegressor(nn, epochs=200, batch_size=10)

    supervised_pipeline = make_pipeline(
        model
    )

    #fold dataset
    kf = KFold(n_splits=3, shuffle=True)
    error_kfold, cv_preds = cv_function(
        supervised_pipeline, kf, X_trans, X_, dist_error)
    print(error_kfold)

    #plot
    # plot_similarity(X_, cv_preds)

    #corcoef
    # print(calculate_corrcoef(X_, cv_preds))

    #knn_percetage_preserved
    knn_perc = knn_percentage_preserved(X_, cv_preds, n=100)
    print('KNN between X_ and cv_preds')
    print(knn_perc, knn_perc.mean(), np.median(knn_perc))

    knn_perc_Xtrans = knn_percentage_preserved(X_trans, cv_preds, n=100)
    print('KNN between X_trans and cv_preds')
    print(knn_perc_Xtrans, knn_perc_Xtrans.mean(), np.median(knn_perc_Xtrans))

    #distances
    # print(pairwise_distances_error(X_, cv_preds))
