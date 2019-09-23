# -*- coding: utf-8 -*-
"""
Created on Fri. Sept. 13 19:33:06 2019

@author: David Stroud
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 19:33:06 2019

@author: David Stroud
"""

# Note: This code does not produce output, rahter it is intended to provide hints
# to a solution for HW1. Good luck!!!!!!

# Hint ~ List of libraries I used to complete the project

# ignore all future warnings
from warnings import simplefilter
import json
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
simplefilter(action='ignore', category=FutureWarning)

# Part 1: Loading Data set - Breast Cancer Data Set within sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
cancer = datasets.load_breast_cancer()

# array M includes the X's/Matrix/the data
M = cancer.data
# M.shape

# Array L includes Y values/labels/target
L = cancer.target
# L.shape[0]

# Enter: Number of folds (k-fold) cross validation
n_folds = 10

# Creating List of Classifiers to use for analysis
clfsList = [RandomForestClassifier, LogisticRegression, KNeighborsClassifier]

# Enter: Range of Hyper-parameters  user wishes to manipulate
# for each classifier listed
# NOTE: No effort was placed on improving Accuracy by manipulating
# hyper-parameters
# Paramters were chosen as examples only.
clfDict = {'RandomForestClassifier': {
    "min_samples_split": [2, 3, 4],
    "n_jobs": [1, 2]},
    'LogisticRegression': {"tol": [0.001, 0.01, 0.1]},
    'KNeighborsClassifier': {
    "n_neighbors": [2, 3, 5, 10, 25],
    "algorithm": ['auto', 'ball_tree', 'brute'],
    "p": [1, 2]}}

''' Sample of documentation for one hyperparameter
Documentation for min_samples_split:
        min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

        If int, then consider min_samples_split as the minimum number.
        If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples)
        are the minimum number of samples for each split.

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
'''

# Pack the arrays together into "data"
data = (M, L, n_folds)

# Printing Out Data Values
# print(data)


# ##------------ run Function - Begin ------------####
#####################################################
# Takes variables a_clf (Classifier function); data (X-data set,
# Y-classification, and, number of folds for CV); and clf_hyper
# (hyper-parameters for classifier)
# Creates folds and .fits model.
# Returns parameters, train/test data, and accuracy score.


def run(a_clf, data, clf_hyper={}):
    M, L, n_folds = data  # EDIT: unpack the "data" container of arrays
    kf = KFold(n_splits=n_folds)  # JS: Establish the cross validation
    ret = {}  # JS: classic explicaiton of results

    # EDIT: We're interating through train and test indexes by using kf.split
    # from M and L.
    # We're simply splitting rows into train and test rows
    # for our five folds.
    for ids, (train_index, test_index) in enumerate(kf.split(M, L)):

        # JS: unpack paramters into clf if they exist
        # EDIT: this gives all keyword arguments except
        clf = a_clf(**clf_hyper)
        #      for those corresponding to a formal parameter
        #      in a dictionary.

        # EDIT: First param, M when subset by "train_index",
        clf.fit(M[train_index], L[train_index])
        #      includes training X's.
        #      Second param, L when subset by "train_index",
        #      includes training Y.

        # EDIT: Using M -our X's- subset by the test_indexes,
        pred = clf.predict(M[test_index])
        #      predict the Y's for the test rows.

        ret[ids] = {'clf': clf,  # EDIT: Create arrays of
                    'train_index': train_index,
                    'test_index': test_index,
                    'accuracy': accuracy_score(L[test_index], pred)
                    }
    return ret

# ##------------ run Function - End   ------------####
#####################################################

# ##-------- myClfHypers Function -Begin --------####
####################################################
# Takes Classifier hyper-parameter dictionary and creates all-possible
# combinations of hyper-parameter

def myClfHypers(clfsList):
    ret_hyper = dict()
    for clf in clfsList:
        clfString = str(clf)  # Check if values in clfsList are in clfDict
        # print("clf: ", clfString)
        for k1, v1 in clfDict.items():  # go through first level of clfDict
            if k1 in clfString:		# if clfString1 matches first level
                ret_hyper[clf] = [dict(zip(v1, s))
                                  for s in product(*v1.values())]
    return ret_hyper

# ##-------- myClfHypers Function - End  --------####
####################################################

# Function Call for parsing hyper parameters from dictionary
hyper_param_dict = myClfHypers(clfsList)
# print(hyper_param_dict)


'''Hint Hint Hint'''

# Function Call for fitting model with given Classifier and hyper-parameter
# combination, using provided Data and n_fold CV, producing a dictionary
# containing Classifier, and corresponding fold accuracies


clfsAccuracyDict = {}
results = {}


# I will stop here and tell you to run the code that you need to write
# a nested for loop. Good luck!!!!!


# for clfs in clfsList:


        # for key in results:
