
import os
import sys
import json
import statistics
from pprint import pprint
from warnings import simplefilter

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

from models.classifiers import svm, rf, logReg, GaussianNB, KNN
from models.grids import Grid, GridSearch, loadGrid

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', lineno=1544)

output_path = "{}/outputs/".format(os.path.dirname(os.path.abspath(__file__)))

text = f'# 1. write a function to take a list or dictionary of clfs and hypers ie use\n\
#    logistic regression, each with 3 different sets of hyper parameters for\n\
#    each\n\
# 2. expand to include larger number of classifiers and hyperparmater settings\n\
# 3. find some simple data\n\
# 4. generate matplotlib plots that will assist in identifying the optimal clf\n\
#    and parampters settings\n\
# 5. Please set up your code to be run and save the results to the directory\n\
#    that its executed from\n\
# 6. Collaborate to get things\n\
# 7. Investigate grid search function'


def part_one(grids):
    return GridSearch(grids)


def part_three():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    n_folds = 5
    n_splits, random_state = 10, 101

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    #                                                    random_state=random_state)

    return dict(X=X, y=y, n_splits=n_splits, random_state=random_state)


def part_four(gs, data):
    # try:
    #     print('*'*20+'\nLOADING STORED MODEL\nTo start fresh move or delete existing pt5.pickle\n'+'*'*20)
    #     gs = loadGrid(output_path+'pt5.pickle')
    # except:
    gs.run(**data)

    gs.vizualizeAll(saveFigsPath=output_path)
    return gs


def part_five(gs):
    gs.save(path=output_path+'pt5.pickle')
    print('GridSearch Object saved at :\n{}'.format(output_path+'pt5.pickle'))
    # gs.pprint(outputPath=output_path+'pt5.txt')


def part_six_seven(gs):
    gs.pprint(outputPath=output_path+'pt6-7.txt')


def main():
    import inspect
    logReg_Grid1 = Grid(
        logReg(),
        grid_opts=[{'C': [0.1, 1, 10],
                    'max_iter': [100, 500, 1000],
                    'penalty': ['l2']}])

    rf_Grid = Grid(
        rf(),
        grid_opts=[{'n_estimators': [10, 100],
                    'max_features': ['auto'],
                    'max_depth': [1, 5, None],
                    'criterion':['gini', 'entropy']}])

    svm_Grid = Grid(
        svm(),
        grid_opts=[{
            'C': [0.01, 0.1, 1, 10],
            'kernel': ['linear', 'poly']
        }, {
            'C': [0.01, 0.1, 1, 10],
            'gamma': [0.0001, 0.001, 0.01, 1],
            'kernel': ['rbf']
        }])

    gnb_Grid = Grid(
        GaussianNB(),
        grid_opts=[{'priors':[None]}])

    knn_Grid = Grid(
        KNN(),
        grid_opts=[{
            'n_neighbors': [3,5,8,10]
        }])


    logReg_Grid = Grid(
        logReg(),
        grid_opts=[{'C': [0.1, 1, 10],
                    'max_iter': [100, 500, 1000],
                    'penalty': ['l2']}])

    grids1 = {
        'logReg': logReg_Grid1
    }
    
    grids2 = {
        # 'rf': rf_Grid,
        'svm': svm_Grid,
        'logReg': logReg_Grid,
        'gausianNB': gnb_Grid,
        'knn': knn_Grid
    }
    resp = {}
    parts = dict(
        pt1=(part_one, grids1),
        pt2=(part_one, grids2),
        pt3=(part_three, False),
        pt4=(part_four, ['pt2', 'pt3']),
        pt5=(part_five, 'pt4'),
        pt67=(part_six_seven, 'pt4')
    )

    print(text)

    for idx, (key, (pt, opts)) in enumerate(parts.items()):

        part_str = '#'*25+'\n# Part {}\n'.format(idx+1)+'#'*25
        print(part_str, '\n')
        if not opts:
            resp[key] = pt()
        elif type(opts) == str:
            opts = resp[opts]
            resp[key] = pt(opts)
        elif type(opts) == list:
            newOpts = []
            for opt in opts:
                if type(opt) == str:
                    newOpts.append(resp[opt])
                else:
                    pass
            resp[key] = pt(*newOpts)
        else:
            resp[key] = pt(opts)
        print('Code Preview:', inspect.getsource(pt), sep='\n')

if __name__ == "__main__":
    main()
