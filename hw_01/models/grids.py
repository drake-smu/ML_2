
import numpy as np
from sklearn.base import clone
import itertools
from pprint import pprint

import pickle

import matplotlib.pyplot as plt

class Grid:
    def __init__(self, model, grid_opts:list):
        self.model = model
        self.grid_opts = []
        self.__hyper_sets = {}
        self.gridSets = {}
        if type(grid_opts)==list:
            self.grid_opts = grid_opts
        elif type(grid_opts)==dict:
            self.grid_opts.append(grid_opts)
        
    def __pre(self):
        hypers = [list(l) for l in [self.__hyperGridGenerator(hs) for hs in self.__gridSetsGenerator()]]
        
    def __gridSetsGenerator(self):
        resp = []
        for (i,grid_set) in enumerate(self.grid_opts):
            self.__hyper_sets[i] = self.__buildHypers(grid_set)
            self.gridSets.update({i:{'params': grid_set,'hypers':[]}})
            resp.append(i)
        return resp
    def __hyperGridGenerator(self, i):
        resp = []
        for ii,hypers in enumerate(self.__hyper_sets[i]):
            self.gridSets[i]['hypers'].append(hypers)
            resp.append(hypers)
        return resp

    def __buildHypers(self, grid_set):
        keys,vals = grid_set.keys(),grid_set.values()
        resp = []
        for instance in itertools.product(*vals):
            resp.append(dict(zip(keys, instance)))
        return resp
    
    def train(self, X, y, n_splits, random_state):
        self.__pre()
        self.max_model = None
        self.max_score = -1000
        self.min_model = None
        self.min_score = 1000
        for (i,gridSet) in self.gridSets.items():
            self.gridSets[i]['results'] = []
            for hypers in gridSet['hypers']:
                _score = 0
                folds = self.model.train(X, y, n_splits, random_state, hypers)
                _score = np.mean([folds[x]['score'] for x in folds.keys()])
                
                if _score > self.max_score:
                    self.max_score, self.max_model = _score, clone(self.model.model).set_params(**hypers)
                if _score < self.min_score:
                    self.min_score, self.min_model = _score, clone(self.model.model).set_params(**hypers)
                self.gridSets[i]['results'].append(folds)
            

    def format_results(self, gridID=0):
        params = []
        scores = []
        means = []
        stds = []
        # pprint(self.gridSets[gridID])
        for hp_id,hp in enumerate(self.gridSets[gridID]['hypers']):
            _params = hp
            # print('params: ',_params)
            _scores = []
            for (fold_id,fold) in self.gridSets[gridID]['results'][hp_id].items():
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
            
        return (params,scores,means, stds)

    def vizualizeSet(self, gridID=0, figSize=(20,6)):
        params, scores, means, stds = self.format_results(gridID)

        fig1,axes = plt.subplots(nrows=1,ncols=2, figsize=figSize)
        
        left_plot = axes[0]
        left_plot.boxplot(scores, patch_artist=True)
        left_plot.set_title("Model Resample Scores Comparison | Model '{}', Grid {}".format(
                        self.model.name, gridID))
        left_plot.set_xlabel('Model #')
        left_plot.set_ylabel('Score')
        # left_plot.set_xticklabels(params,
        #             rotation=65, fontsize=5)
        right_plot = axes[1]
        right_plot.scatter(means, stds, alpha=0.5)
        right_plot.set_title(
                "Bias / Variance Analysis | Model '{}', Grid {}".format(
                        self.model.name, gridID))
        right_plot.set_xlabel('Mean Resample Scores per Model')
        right_plot.set_ylabel('Std of Resample Scores per Model')
        plt.savefig("plots/'{}'_'{}'.png".format(self.model.name, gridID))
        

    def vizualizeFull(self):
        for id in self.gridSets.keys():
            self.vizualizeSet(gridID=id)

    def pprint(self):
        for id in self.gridSets.keys():
            params, scores, means, stds = self.format_results(id)
            print("-"*50)
            print("Model '{}', Grid {}".format(self.model.name, id))
            print("-"*50)
            for index, (mean, std, _params) in enumerate(zip(means, stds, params)):
                if mean==self.max_score:
                    print("Model max+  {} >>  Score: {:>5.3f} (+/-{:>5.3f} 95% CI) | Parameters: {}".format(index+1, mean, std * 2, _params))
                elif mean==self.min_score:
                    print("Model min-  {} >>  Score: {:>5.3f} (+/-{:>5.3f} 95% CI) | Parameters: {}".format(index+1, mean, std * 2, _params))
                else:
                    print("Model       {} >>  Score: {:>5.3f} (+/-{:>5.3f} 95% CI) | Parameters: {}".format(index+1, mean, std * 2, _params))
            print()
        
        


class GridSearch:
    def __init__(self, grids, X, y, n_splits, random_state):
        self.X = X
        self.y = y
        self.grids = grids
        self.n_splits = n_splits
        self.random_state = random_state
        self.results = {}
        self.max_model = None
        self.max_score = -1000
        self.min_model = None
        self.min_score = 1000
        self.training = (X, y, n_splits, random_state)

    def run(self):
        # print("RUNNING SEARCH")
        test = []
        for (name,grid) in self.grids.items():
            print(name,'...')
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
        return test
    
    def save(self,path="results.pickle"):
        with open(path, 'wb') as f:
            
            pickle.dump(self,f)

    def vizualizeAll(self):
        for grid in self.grids.values():
            grid.vizualizeFull()

    def pprint(self):
        for grid in self.grids.values():
            grid.pprint()
        print("-"*50)
        print("Max Model '{}'\n'{}'\n Score {}".format(self.max_name,self.max_model, self.max_score))
        print("Min Model '{}'\n'{}'\n Score {}".format(self.min_name,self.min_model, self.min_score))
        print("-"*50)
def loadGrid(path="results.pickle"):
    with open(path, 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
        data = pickle.load(f)
        return data


