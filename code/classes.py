from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

import sys
import time


class NNReplicator(TransformerMixin):

    def __init__(self,
                 embedder,
                 layers, dropout, lr, act_func, loss_func, epochs, batch_size):

        self.embedder = embedder
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        self.act_func = act_func
        self.loss_func = loss_func
        self.epochs = epochs
        self.batch_size = batch_size

    def nnConstruct(self, shape):

        model = Sequential()

        for i, (layer, drop) in enumerate(zip(self.layers, self.dropout)):

            if i == 0:
                model.add(Dense(layer, input_shape=(
                    shape,), activation=self.act_func))
            else:
                model.add(Dense(layer, activation=self.act_func))

            model.add(Dropout(drop))

        model.add(Dense(self.embedder.n_components, activation='linear'))
        ada = optimizers.Adagrad(lr=self.lr)

        model.compile(optimizer=ada, loss=self.loss_func)

        self.krObject = KerasRegressor(
            lambda: model, epochs=self.epochs, batch_size=self.batch_size)

    def fit(self, X, y=None):

        shape = X.shape[1]
        self.nnConstruct(shape)

        X_ = self.embedder.fit_transform(X)

        self.krObject.fit(X, X_)
        return self

    def transform(self, X):
        return self.krObject.predict(X)


class NNPipeline(Pipeline):

    def __init__(self, steps, memory=None):
        super(NNPipeline, self).__init__(
            steps=steps, memory=memory)

        assert isinstance(self.steps[-1][1], NearestNeighbors)

    def transform(self, X):
        X_ = X.copy()
        for step in self.steps[:-1]:
            X_ = step[1].transform(X_)
        return X_

    def kneighbors(self, X, n_neighbors, return_distance):

        # Finally, run kneighbors method using the last step (NN)
        return self.steps[-1][1].kneighbors(
            X=self.transform(X), n_neighbors=n_neighbors,
            return_distance=return_distance
        )

    def get_memory_usage(self, X):

        return sys.getsizeof(self.transform(X))

    def get_query_time(self, X, n_neighbors, return_distance, number=100):

        X_ = self.transform(X)
        time_func = lambda: self.steps[-1][1].kneighbors(
            X=X_, n_neighbors=n_neighbors,
            return_distance=return_distance
        )
        return timeit.timeit(time_func, number=number) / number
