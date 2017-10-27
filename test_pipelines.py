import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.ensemble  import RandomForestClassifier
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer

from ReplicatorClass import NNReplicator
from nnlearn import dist_loss
from nnlearn import cv_function
from nnlearn import load_bankruptcy
from nnlearn import load_wines
from nnlearn import load_page

from lightgbm import LGBMClassifier

if __name__ == '__main__':

    #dataset
    # X, y = load_bankruptcy('datasets/bankruptcy.data')
    # X, y = load_wines('datasets/winequality-white.csv')
    X, y = load_page('datasets/spambase.data')
    # dataset = load_breast_cancer()
    # X, y = dataset['data'], dataset['target']

    #pipeline number 1

    model1 = make_pipeline(
        Imputer(),
        LGBMClassifier(
            n_estimators=500, learning_rate=0.01, num_leaves=20, subsample=1, colsample_bytree=0.8)
        )

    #pipeline number 2

    model2 = make_pipeline(
        Imputer(),
        StandardScaler(),
        PCA(n_components=2),
        LGBMClassifier(
            n_estimators=500, learning_rate=0.01, num_leaves=20, subsample=1, colsample_bytree=0.8)
        )

    #pipeline number 3
    tsne = TSNE(n_components=2, verbose=10)

    model3 = make_pipeline(
        Imputer(),
        StandardScaler(),
        NNReplicator(
            2, tsne, [1000], [0], 0.1, 'sigmoid', dist_loss, 200, 10),
        LGBMClassifier(
            n_estimators=1000, learning_rate=0.01, num_leaves=25, subsample=0.8, colsample_bytree=0.8)
        )

    #create loop of models

    totalpipes = [model1, model2, model3]

    kf = StratifiedKFold(n_splits=10, shuffle=True)

    error_kfold, cv_preds = cv_function(model3, kf, X, y, roc_auc_score)
    print(error_kfold)

    # for i,(pipe) in enumerate(totalpipes):

    #     print("Model number: " + str(i))
    #     error_kfold, cv_preds = cv_function(pipe, kf, X, y, roc_auc_score)
    #     print(error_kfold)
