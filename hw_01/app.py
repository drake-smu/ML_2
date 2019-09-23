
from warnings import simplefilter
import json
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
simplefilter(action='ignore', category=FutureWarning)

from classifiers.logistic_reg import logisticReg as LogisticRegression

cancer = datasets.load_breast_cancer()

# array M includes the X's/Matrix/the data
M = cancer.data
# M.shape

# Array L includes Y values/labels/target
L = cancer.target
# L.shape[0]

# Enter: Number of folds (k-fold) cross validation
n_folds = 10

clfsList = [LogisticRegression]

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    n_folds = 5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=101)

    # Support for:
    # (1) Multiple model types and
    # (2) multiple independent grids for each model
    grids = {'LR': {'model': LogisticRegression(solver='lbfgs',
                                             multi_class='auto'),
                    'param_grid': [{'C': [1, 10],
                                    'max_iter': [500, 1000],
                                    'penalty': ['l2']}]}
                                    }

# Pack the arrays together into "data"
data = (M, L, n_folds)

# Printing Out Data Values
print(data)