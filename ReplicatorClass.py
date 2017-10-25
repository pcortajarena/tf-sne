from sklearn.base import TransformerMixin
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor


class NNReplicator(TransformerMixin):

    def __init__(self, n_comp, embedder, layers, dropout, lr, act_func, loss_func, epochs, batch_size):

        self.n_comp = n_comp
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
                model.add(Dense(layer, input_shape=(shape,), activation=self.act_func))
            else:
                model.add(Dense(layer, activation=self.act_func))

            model.add(Dropout(drop))

        model.add(Dense(self.n_comp, activation='linear'))
        ada = optimizers.Adagrar(lr=self.lr)

        model.compile(optimizer=ada, loss=self.loss_func)

        self.krObject = KerasRegressor(model, epochs=self.epochs, batch_size=self.batch_size)


    def fit(self, X):

        shape = X.shape[1]
        neural_net = nnConstruct(shape)

        X_ = self.embedder.fit_transform(X)

        self.krObject.fit(X, X_)


    def transform(self, X):

        self.krObject.predict(X)
