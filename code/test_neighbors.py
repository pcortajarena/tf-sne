
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from classes import NNReplicator, NNPipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
#datasets
from utils import load_page, load_bankruptcy
from utils import dist_loss
#numpy
import numpy as np

import pdb


#function to see neighbors preserved
def devidedisin(x, split=None):
    if not split:
        split = int(x.shape[0]/2)
    return np.isin(x[split:], x[0:split])

def nn_preserved(y_true, y_pred, n_true, n_pred):
    conc = np.hstack((y_pred, y_true))
    return np.apply_along_axis(devidedisin, 1, conc, split=n_pred).mean(axis=1)


def compare(X_testing, pipe1, pipe2, k1, k2):

    neighbors1 = pipe1.kneighbors(X_testing, n_neighbors=k1, return_distance=False)
    neighbors2 = pipe2.kneighbors(X_testing, n_neighbors=k2, return_distance=False)

    return nn_preserved(neighbors2, neighbors1, k2, k1)

#main
if __name__ == '__main__':

    #dataset
    X, y = load_bankruptcy('../datasets/bankruptcy.data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    #pipeline 1 = PRED
    md = MDS(n_components = 4, random_state = 0, n_jobs = -1, verbose = 10, n_init = 4)
    nnreplicator = NNReplicator(4, md, [1000], [0], 0.1, 'sigmoid', dist_loss, 200, 10)

    pipeline1 = NNPipeline([
        ('imp', Imputer()),
        ('std', StandardScaler()),
        ('nnrep', nnreplicator),
        ('knn', NearestNeighbors(algorithm='brute'))])

    #training
    pipeline1.fit(X_train)

    #pipeline 2 = TRUE
    pipeline2 = NNPipeline([
        ('imp', Imputer()),
        ('std', StandardScaler()),
        ('knn', NearestNeighbors(algorithm='brute'))])

    #training
    pipeline2.fit(X_train)

    preserved = compare(X_test, pipeline1, pipeline2, k1 = 100, k2 = 50)







