
import sys
import json
import statistics
from pprint import pprint
from warnings import simplefilter

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

from models.classifiers import svm, rf, logReg
from models.grids import Grid, GridSearch, loadGrid

simplefilter(action='ignore', category=FutureWarning)


# 1. write a function to take a list or dictionary of clfs and hypers ie use
#    logistic regression, each with 3 different sets of hyper parameters for
#    each
# 2. expand to include larger number of classifiers and hyperparmater settings
# 3. find some simple data
# 4. generate matplotlib plots that will assist in identifying the optimal clf
#    and parampters settings
# 5. Please set up your code to be run and save the results to the directory
#    that its executed from
# 6. Collaborate to get things
# 7. Investigate grid search function



def testClassifiers():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    n_folds = 5
    n_splits, random_state = 10,101

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=101)
    
    rf_Grid = Grid(rf(), 
        grid_opts=[{'n_estimators': [10, 100],
        'max_features': ['auto'],
        'max_depth': [1,5],
        'criterion':['gini', 'entropy']}])
    
    svm_Grid = Grid(svm(), 
        grid_opts=[{
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear','poly']
        },{
            'C': [0.01, 0.1, 1, 10],
            'gamma': [0.0001, 0.001, 0.01, 1],
            'kernel': ['rbf']
            }])

    logReg_Grid = Grid(logReg(), 
        grid_opts=[{'C': [1, 10],
        'max_iter': [500, 1000],
        'penalty': ['l2']}])

    grids = {
        'rf': rf_Grid,
        'svm': svm_Grid,
        'logReg': logReg_Grid
    }
    

    gs = GridSearch(grids,X_train, y_train, n_splits, random_state)
    gs.run()
    gs.save()
    return gs

def testLoad():
    gs = loadGrid()
    return gs

if __name__ == "__main__":
    print("#"*25,'  Starting Classifier Optimization...', "#"*25, sep='\n', end='\n')
    # gs = testClassifiers()
    gs = testLoad()

    
    gs.vizualizeAll()
    orig_stdout = sys.stdout
    f = open('out.txt', 'w')
    sys.stdout = f
    gs.pprint()
    print("See `plots/` for visualizations.")
    sys.stdout = orig_stdout
    f.close()
    
