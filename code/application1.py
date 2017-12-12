import pandas as pd
import numpy as np

from seaborn import heatmap
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
import keras.backend as K

#distance loss
def dist_loss(y, y_):
    return K.mean(K.sqrt(K.sum(K.square(y - y_), axis=1)))

def dist_error(y_true, y_pred):
    return np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)))

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

def app(unsuper_pipe, super_pipe, X, y):

    X_unsupervised = unsuper_pipe.fit_transform(X)

    X_train, X_test, X_unsuper_train, X_unsuper_test, y_train, y_test = train_test_split(X, X_unsupervised, y)

    super_pipe.fit(X_train, X_unsuper_train)

    predictions = super_pipe.predict(X_test)

    return predictions, X_unsuper_test, y_test


if __name__ == '__main__':
    
    dataset = load_digits()

    X, y = dataset['data'], dataset['target']

    tsne = TSNE(n_components=2, verbose=10)

    #pipelines
    unsupervised_pipeline = make_pipeline(
        StandardScaler(),
        tsne
    )

    nn = lambda: reg(
        X.shape[1], loss_func=dist_loss,
        layers=[1000], dropout=[0],
        output_shape=2, lr=0.1, act_function='sigmoid')

    model = KerasRegressor(nn, epochs=200, batch_size=10)

    supervised_pipeline = make_pipeline(
        StandardScaler(),
        model
    )

    total_predictions, X_tsne_test, y_test = app(unsupervised_pipeline, supervised_pipeline, X, y)

    #example of number representation
    fig, ax = plt.subplots(1,1, figsize=(9,8))
    heatmap(X[550].reshape(8,8), cmap='gray', annot=True, ax=ax, cbar=False)
    plt.axis('off')
    plt.savefig('../text/figures/exampledigit.pdf', bbox_inches='tight')

    error = dist_error(X_tsne_test.iloc[:,:2].values, total_predictions.iloc[:,:2].values)




