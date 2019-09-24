
import numpy as np
import itertools
from pprint import pprint

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone


class classifier:
    """
    Base Class for wrapping classification models. Specific
    classification models inherit this.
    """

    def __init__(self, model, params={}):
        self.model_type = 'classifier'
        self.model = model(**params)

    def train(self, X, y, n_splits, random_state, hypers):
        """
        Method to train model object using given training data and parameters

        Arguments:
            X {matrix} -- Data for training features
            y {array} -- Array of expected response values
            n_splits {int} -- Number of splits for construncting K-folds CV
            random_state {int} -- Random seed
            hypers {dict} -- dictionary of hyperparams. These vary per model
            type. Check sklearn docs.

        Returns:
            dict -- Returns dictionary result dictionaries for each fold
            evaluation. Each fold result contains the model, training_index,
            test_index, and accuracy score.
        """
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=random_state)
        resp = {}
        for idx, (train_index, test_index) in enumerate(kf.split(X, y)):
            model = clone(self.model)
            model.set_params(**hypers)
            model.fit(X[train_index], y[train_index])
            pred = model.predict(X[test_index])
            score = accuracy_score(y[test_index], pred)

            sample_name = 'fold_' + str(idx)
            resp[sample_name] = {'model': model,
                                 'train_index': train_index,
                                 'test_index': test_index,
                                 'score': score}
        return resp


class logReg(classifier):
    """Wrapper for Logistic Regression Models
    
    Arguments:
        classifier {[type]} -- [description]
    """
    def __init__(self):
        params = dict(penalty="none",
                      dual=False, tol=0.0001,
                      C=1, fit_intercept=True,
                      intercept_scaling=1, class_weight=None,
                      random_state=None,
                      max_iter=100, multi_class='ovr',
                      verbose=0, n_jobs=-1, l1_ratio=None)
        self.name = "Logistic Regression"
        self.filename = 'logistic_regression'
        super().__init__(model=LogisticRegression, params=params)


class svm(classifier):
    """
    Wrapper for SVM Models

    Arguments:
        None
    """
    def __init__(self):
        self.name = "SVM"
        self.filename = 'svm'
        super().__init__(model=SVC)


class rf(classifier):
    """Wrapper for Random Forest Models
    
    Arguments:
        classifier {[type]} -- [description]
    """
    def __init__(self):
        params = dict(
            n_jobs=-1
        )
        self.name = "Random Forest"
        self.filename = 'random_forest'
        super().__init__(RandomForestClassifier, params=params)
