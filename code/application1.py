import pandas as pd
from scipy import sparse
from sklearn.datasets import load_digits

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline

if __name__ == '__main__':
    
    dataset = load_digits()

    X, y = dataset['data'], dataset['target']

    #calculate tsne

    tsne = TSNE(n_components=2, verbose=10)

    #pipelines
    
    trans_pipeline = make_pipeline(
        StandardScaler(with_mean=False)
        )

    unsupervised_pipeline = make_pipeline(
        tsne
    )

    #step 1: transformation and aplying tsne

    X_trans = trans_pipeline.fit_transform(X)

    X_ = unsupervised_pipeline.fit_transform(X_trans)