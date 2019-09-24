
import sys
import numpy as np
from sklearn.base import clone
import itertools
from pprint import pprint

import pickle

import matplotlib.pyplot as plt


class Grid:
    """
    A class used to train an analyze a given modal over
    a specified hyperparameters.
    ...

    Attributes
    ----------
    model : Object
        model is a model object from the wrapper module `models`
    grid_opts : list
        grid_opts are a list (or dict) of hyperparamter options.
        the options are used to generate all possible permutations.
        Multiple options can be used to cut down on total hyper params
        tested.
    gridSets : dict
        gridSets contains the results for each gridSet. A grid set is
        built using a single grid_option element and contains all
        associated hyperparams for that set.
    num_legs : int
        the number of legs the animal has (default 4)
    max_model : object
        contains the model object for the model configuration
        that scored highest on accuracy
    max_score : float
        the mean accuracy score of the max_model
    min_model : object
        same as max but for the worst performing model
    min_score : float
        the mean accuracy score of the min_model

    Methods
    -------
    train(X, y, n_splits, random_state)
        Trains the grid of models using X training features and
        y resp data.
    vizualizeSet(gridID=0, figSize=(20, 6))
        Generates matplotlib plts for accuracy results and bias for
        models of specified hyper parameter set.
    vizualizeFull(saveFigsPath=False)
        Iterates over all gridSets performs vizualizeSet on each.
        saveFigsPath is an optional param to save figs to output
        files.
    pprint(outputPath=False)
        Print trained results for all models in Grid. Optional
        path to write output to file instead of std.out.

    """

    def __init__(self, model, grid_opts: list):
        self.model = model
        self.grid_opts = []
        self.__hyper_sets = {}
        self.gridSets = {}
        self.max_model = None
        self.max_score = -1000
        self.min_model = None
        self.min_score = 1000
        if type(grid_opts) == list:
            self.grid_opts = grid_opts
        elif type(grid_opts) == dict:
            self.grid_opts.append(grid_opts)

    def __pre(self):
        hypers = [list(l) for l in [self.__hyperGridGenerator(hs)
                                    for hs in self.__gridSetsGenerator()]]

    def __gridSetsGenerator(self):
        resp = []
        for (i, grid_set) in enumerate(self.grid_opts):
            self.__hyper_sets[i] = self.__buildHypers(grid_set)
            self.gridSets.update({i: {'params': grid_set, 'hypers': []}})
            resp.append(i)
        return resp

    def __hyperGridGenerator(self, i):
        resp = []
        for ii, hypers in enumerate(self.__hyper_sets[i]):
            self.gridSets[i]['hypers'].append(hypers)
            resp.append(hypers)
        return resp

    def __buildHypers(self, grid_set):
        keys, vals = grid_set.keys(), grid_set.values()
        resp = []
        for instance in itertools.product(*vals):
            resp.append(dict(zip(keys, instance)))
        return resp

    def train(self, X, y, n_splits, random_state):
        self.__pre()
        for (i, gridSet) in self.gridSets.items():
            self.gridSets[i]['results'] = []
            for hypers in gridSet['hypers']:
                _score = 0
                folds = self.model.train(X, y, n_splits, random_state, hypers)
                _score = np.mean([folds[x]['score'] for x in folds.keys()])

                if _score > self.max_score:
                    self.max_score, self.max_model = _score, clone(
                        self.model.model).set_params(**hypers)
                if _score < self.min_score:
                    self.min_score, self.min_model = _score, clone(
                        self.model.model).set_params(**hypers)
                self.gridSets[i]['results'].append(folds)

    def format_results(self, gridID=0):
        params = []
        scores = []
        means = []
        stds = []
        # pprint(self.gridSets[gridID])
        for hp_id, hp in enumerate(self.gridSets[gridID]['hypers']):
            _params = hp
            # print('params: ',_params)
            _scores = []
            for (fold_id, fold) in self.gridSets[gridID]['results'][hp_id].items():
                _score = fold['score']
                # print('score: ',_score)
                _scores.append(_score)
            _mean_score = np.mean(_scores)
            _std_score = np.std(_scores)
            # print(_scores)
            params.append(_params)
            scores.append(_scores)
            means.append(_mean_score)
            stds.append(_std_score)

        return (params, scores, means, stds)

    def vizualizeSet(self,
                     gridID=0,
                     figSize=(20, 6)):
        params, scores, means, stds = self.format_results(gridID)

        fig1, (left_plot, right_plot) = plt.subplots(
            nrows=1, ncols=2, figsize=figSize)
        fig1.suptitle("Model '{}', Grid {}".format(
            self.model.name, gridID))

        left_plot.boxplot(scores, patch_artist=True)
        left_plot.set_title("Model Resample Scores Comparison")
        left_plot.set_xlabel('Model #')
        left_plot.set_ylabel('Score')

        right_plot.scatter(means, stds, alpha=0.5)
        right_plot.set_title(
            "Bias / Variance Analysis")
        right_plot.set_xlabel('Mean Resample Scores per Model')
        right_plot.set_ylabel('Std of Resample Scores per Model')
        return fig1

    def vizualizeFull(self,
                      saveFigsPath=False):
        figures = []
        for grid_id in self.gridSets.keys():
            _fig = self.vizualizeSet(gridID=grid_id)
            if saveFigsPath:
                _fig.savefig(
                    "{}plots/{}_{}.png".format(saveFigsPath, self.model.name, grid_id))
            figures.append(_fig)
        return figures

    def pprint(self, outputPath=False):
        for id in self.gridSets.keys():
            params, scores, means, stds = self.format_results(id)

            print("="*50,
                  "Model '{}', Grid {}".format(self.model.name, id),
                  "-"*50,
                  sep="\n")

            for index, (mean, std, _params) in enumerate(zip(means, stds, params)):
                line_out = "Model {}\n    Parameters: {}\n    Score: {:>5.3f} (+/-{:>5.3f} 95% CI)".format(
                    index+1, _params, mean, std * 2)
                if mean == self.max_score:
                    line_out = "Model {} [++++]\n    Parameters: {}\n    Score: {:>5.3f} (+/-{:>5.3f} 95% CI)".format(
                        index+1, _params, mean, std * 2)
                elif mean == self.min_score:
                    line_out = "Model {} [----]\n    Parameters: {}\n    Score: {:>5.3f} (+/-{:>5.3f} 95% CI) ".format(
                        index+1, _params, mean, std * 2)
                print(line_out, end='\n\n')
        print("="*50)


class GridSearch:
    """
    GridSearch is a collection of Grid class objects.


    Methods
    -------
    run(X, y, n_splits, random_state)
        Train all Grids in collection on same data and
        settings.
    save(path="results.pickle")
        Store serialized GridSearch object to pickle file.
    vizualizeAll(saveFigsPath=False)
        Iterates over all Grids and generates vizualizations
        for all their gridsets.
    pprint(outputPath=False)
        Print trained results for all models in Grid. Optional
        path to write output to file instead of std.out.
    """
    def __init__(self, grids):
        self.grids = grids
        self.results = {}
        self.max_model = None
        self.max_score = -1000
        self.min_model = None
        self.min_score = 1000
        self.has_run = False

    def run(self, X, y, n_splits, random_state):
        self.n_splits = n_splits
        self.random_state = random_state
        self.X = X
        self.y = y
        self.training = (X, y, n_splits, random_state)
        test = []
        for (name, grid) in self.grids.items():
            print(name, '...')
            grid.train(*self.training)
            _results = grid.gridSets
            grid.vizualizeFull()
            test.append(grid.train(*self.training))
            self.results[name] = _results
            if grid.max_score > self.max_score:
                self.max_score, self.max_model = grid.max_score, grid.max_model
                self.max_name = grid.model.name
            if grid.min_score < self.min_score:
                self.min_score, self.min_model = grid.min_score, grid.min_model
                self.min_name = grid.model.name
        self.has_run = True
        return test

    def save(self, path="results.pickle"):
        with open(path, 'wb') as f:

            pickle.dump(self, f)

    def vizualizeAll(self, saveFigsPath=False):
        for grid in self.grids.values():
            grid.vizualizeFull(saveFigsPath)

    def pprint(self, outputPath=False):

        if outputPath:
            orig_stdout = sys.stdout
            f = open(outputPath, 'w')
            sys.stdout = f

        for grid in self.grids.values():
            grid.pprint()
        print("-"*50)
        print("Max Model '{}'\n'{}'\n Score {}".format(
            self.max_name, self.max_model, self.max_score))
        print("Min Model '{}'\n'{}'\n Score {}".format(
            self.min_name, self.min_model, self.min_score))
        print("-"*50)
        if outputPath:
            sys.stdout = orig_stdout
            f.close()


def loadGrid(path="outputs/results.pickle"):
    """Load GridSearch Object from serialized
    pickle file.

    Keyword Arguments:
        path {str} -- Filepath to target pickle file
        (default: {"outputs/results.pickle"})

    Returns:
        object -- Returns stored GridSearch Object
    """
    with open(path, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
        return data
