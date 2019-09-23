import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from random import choices, sample
import itertools
import math
from multiprocessing import Process, Queue
#import sklearn.datasets
from operator import itemgetter as  itemget
import scipy.stats as ss

class logisticReg:
    def __init__(self, penalty = "none",
                 dual = False, tol = 0.0001, C = 1, fit_intercept = True,
                 intercept_scaling = 1, class_weight = None,
                 random_state = None, solver = 'liblinear', max_iter = 100,
                 multi_class = 'ovr', verbose = 0, n_jobs = -1, l1_ratio =  None):
        self.model = LogisticRegression(penalty = penalty,
                                            dual = dual,
                                            tol = tol,
                                            C = C,
                                            fit_intercept = fit_intercept,
                                            intercept_scaling = intercept_scaling,
                                            class_weight = class_weight,
                                            random_state = random_state,
                                            solver = solver,
                                            max_iter = max_iter,
                                            multi_class = multi_class,
                                            verbose = verbose,
                                            n_jobs = n_jobs,
                                            l1_ratio = l1_ratio
                     )
    def __call__(self):
        print(self.model)
