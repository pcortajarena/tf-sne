from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import timeit
# datasets
from utils import load_page, load_bankruptcy
from utils import dist_loss
# numpy
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import sys
import pickle
import os
import tempfile

import timeit
from joblib import Parallel, delayed

# function to see neighbors preserved


def devidedisin(x, split=None):
    if not split:
        split = int(x.shape[0]/2)
    return np.isin(x[split:], x[0:split])


def nn_preserved(y_true, y_pred, n_true, n_pred):
    conc = np.hstack((y_pred, y_true))
    return np.apply_along_axis(devidedisin, 1, conc, split=n_pred).mean(axis=1)


def compare(X_testing, pipe1, pipe2, k1, k2):

    neighbors1 = pipe1.kneighbors(
        X_testing, n_neighbors=k1, return_distance=False)

    neighbors2 = pipe2.kneighbors(
        X_testing, n_neighbors=k2, return_distance=False)

    timenn = pipe1.get_query_time(
        X_testing, n_neighbors=1000, return_distance=False)

    timereal = pipe2.get_query_time(
        X_testing, n_neighbors=1000, return_distance=False)

    return nn_preserved(neighbors2, neighbors1, k2, k1), timenn, timereal


def get_memory_object(obj):
    fname = tempfile.mktemp()
    pickle.dump(obj, open(fname, 'wb'))
    size = os.path.getsize(fname)
    os.remove(fname)
    return size


def modelsjob(name, params):

    nnreplicator = NNReplicator(
        md, params['layers'], params['dropout'],
        0.1, 'sigmoid', dist_loss, 200, 10)

    pipeline1 = NNPipeline([
        ('imp', Imputer()),
        ('std', StandardScaler()),
        ('nnrep', nnreplicator),
        ('knn', NearestNeighbors(algorithm='brute'))])

    # training
    pipeline1.fit(X_train)
    mem_model = pipeline1.get_memory_usage(X_train)

    # pipeline 2 = TRUE
    pipeline2 = NNPipeline([
        ('imp', Imputer()),
        ('std', StandardScaler()),
        ('knn', NearestNeighbors(algorithm='brute'))])

    # training
    pipeline2.fit(X_train)
    mem_real = pipeline2.get_memory_usage(X_train)

    preserved, time_model, time_real = compare(
        X_test, pipeline1, pipeline2, k1=100, k2=100)

    return np.mean(preserved), time_model, time_real, mem_model, mem_real

# main
if __name__ == '__main__':

    # Set tf to one thread
    # config = tf.ConfigProto(
    #     intra_op_parallelism_threads=1,
    #     inter_op_parallelism_threads=1,
    #     allow_soft_placement=True,
    #     device_count={'CPU': 1})
    # session = tf.Session(config=config)
    # K.set_session(session)

    # dataset
    data = pd.read_csv('../datasets/blog.csv')

    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # pipeline 1 = PRED
    md = MDS(n_components=100, random_state=0, n_jobs=1,
             verbose=10, n_init=1)

    models = {
        "model1": {
            "layers": [32, 16, 16, 8],
            "dropout": [0, 0, 0, 0],
            "type": 'deep - narrow'
        },
        "model2": {
            "layers": [32, 32],
            "dropout": [0, 0],
            "type": 'shallow - narrow'
        },
        "model3": {
            "layers": [32, 16],
            "dropout": [0, 0],
            "type": 'shallow - narrow'
        },
        "model4": {
            "layers": [128, 64, 64],
            "dropout": [0, 0, 0],
            "type": 'deep - wide'
        },
        "model5": {
            "layers": [128, 64, 64, 32],
            "dropout": [0, 0, 0, 0],
            "type": 'deep - wide'
        },
        "model6": {
            "layers": [128, 128, 128],
            "dropout": [0, 0, 0],
            "type": 'shallow - wide'
        },
        "model7": {
            "layers": [128, 64],
            "dropout": [0, 0],
            "type": 'shallow - wide'
        },
        "model8": {
            "layers": [1000],
            "dropout": [0],
            "type": 'shallow - wider'
        }
    }

    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(modelsjob)(model_name, model_params)
        for model_name, model_params in models.items()
    )

    resultsdf = pd.DataFrame(
        results,
        columns=['npreserved', 'nntime', 'realtime', 'mem_model', 'mem_real'],
        index=models)
    resultsdf.to_latex(buf='../figures/app2aproxbrute100100blog.tex')
