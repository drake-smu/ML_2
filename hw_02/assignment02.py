#%%

import os
import sys

# from numpy import genfromtxt
import numpy as np
import re
import csv
from pprint import pprint

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split

from warnings import simplefilter
from models.classifiers import svm, rf, logReg, GaussianNB, KNN
from models.grids import Grid, GridSearch, loadGrid
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', lineno=1544)

#%%
path = os.path.dirname(os.path.abspath(__file__))
output_path = "{}/outputs/".format(path)
text0 = '''\
A medical claim is denoted by a claim number ('Claim.Number'). 
Each claim consists of one or more medical lines denoted by a claim line number 
('Claim.Line.Number').
'''

text1 = '''\
1. J-codes are procedure codes that start with the letter 'J'.

     A. Find the number of claim lines that have J-codes.

     B. How much was paid for J-codes to providers for 'in network' claims?

     C. What are the top five J-codes based on the payment to providers?
'''

text2 = '''\
2. For the following exercises, determine the number of providers that were paid
for at least one J-code. Use the J-code claims for these providers to complete
the following exercises.

     A. Create a scatter plot that displays the number of unpaid claims (lines
     where the ‘Provider.Payment.Amount’ field is equal to zero) for each
     provider versus the number of paid claims.

     B. What insights can you suggest from the graph?

     C. Based on the graph, is the behavior of any of the providers concerning?
     Explain.
'''

text3 = '''\
3. Consider all claim lines with a J-code.

    A. What percentage of J-code claim lines were unpaid?

    B. Create a model to predict when a J-code is unpaid. Explain why you
    choose the modeling approach.

    C. How accurate is your model at predicting unpaid claims?

    D. What data attributes are predominately influencing the rate of
    non-payment?
'''
texts = (print(t) for t in [text0, text1, text2, text3])

next(texts)



# %%

next(texts)
with open(path+'/claim.sample.csv') as file:
    reader = csv.reader(file)
    l = list(map(list, reader))
    

claims_lables = l[0]
claims_data = np.array(l[1:])
# %%
# 1.a
jCodes = claims_data[[i for i,x in enumerate(claims_data) if re.match("\A(j|J)",x[claims_lables.index('Procedure.Code')])],:]
jCodeRankings = np.asarray(np.unique(jCodes[:,9], return_counts=True)).T
# %%
inOuts,inOutCounts= np.unique(jCodes[:,14], return_counts=True)
inOutRanks = np.asarray((inOuts,inOutCounts)).T


# %%
# inNetworkJ = [x for i,x in enumerate(jCodes) if x[14]=='I']
#1.b
inNetworkJ = jCodes[[i for i,x in enumerate(jCodes) if x[14]=='I']]
sum([float(x[claims_lables.index('Provider.Payment.Amount')]) for x in inNetworkJ])

# %%
# 1.c
jCodeCosts = {}
for entry in jCodes:
    if(entry[9] in jCodeCosts.keys()):
        jCodeCosts[entry[9]].append(float(entry[claims_lables.index('Provider.Payment.Amount')]))
    else:
        jCodeCosts[entry[9]]= [float(entry[claims_lables.index('Provider.Payment.Amount')])]

def sortCosts(x):
    return x[1]
    
jCostList = [(i,round(sum(x),2)) for i,x in jCodeCosts.items()]
jCostRanked = sorted(jCostList,key=sortCosts, reverse=True)




# %%
# Part 2
next(texts)
def _processClaim(prov,claim):
    
    if(float(claim[claims_lables.index('Provider.Payment.Amount')]) > 0):
        # prov['paid'].extend(claim)
        prov['paid'].append(claim);
    else:
        prov['unpaid'].append(claim);
    
    prov['claims'].append(claim)

    return prov

def _addProv(claim):
    
    prov = { 
        'paid': [],
        'unpaid': [],
        'claims': []
        }
    
    return _processClaim(prov,claim)

def updateProviders(providers, claim):
    providerId = claim[4]
    claim[claims_lables.index('Provider.Payment.Amount')] = float(claim[claims_lables.index('Provider.Payment.Amount')])
    if(providerId in providers.keys()):
        providers[providerId] = _processClaim(providers[providerId],claim)
    else:
        providers[providerId] = _addProv(claim)
    return providers
#Lets organize claims into a provider dictionary 
providerClaims = {}

for claim in jCodes:
    providerClaims=updateProviders(providerClaims,claim.tolist())

# Build plot data
plotData = [];
for prov,data in providerClaims.items():
    resp = {}
    resp['provId'] = prov;
    resp['countUnpaid'] = len(data['unpaid']);
    resp['countPaid'] = len(data['paid']);
    resp['countTotal'] = len(data['claims']);
    if((resp['countUnpaid']+resp['countPaid']!=len(data['claims']))):
        print('Whats up here...\n',prov,resp['countUnpaid']+resp['countPaid'],len(data['claims']));

    plotData.append(resp)


# %%
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
# from matplotlib import style as mpstyle
with plt.style.context('seaborn-notebook'):
    scaleT = max([x['countTotal'] for i,x in enumerate(plotData)])
    plotColors = cm.rainbow(len(plotData))
    # plt.cm =plotColors
    fig, ax = plt.subplots()
    # ax.set_facecolor('white')
    for prov in plotData:
        x, y = (prov['countPaid'],prov['countUnpaid'])

        scale = 10+200* prov['countTotal'] / scaleT;
        ax.scatter(x, y,
        s = scale,
        cmap='cool',
        label=prov['provId'],
        alpha=0.8,
        edgecolors='none')
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    for p in [p for p in plotData if p['countUnpaid']>3*p['countPaid'] and p['countTotal']>1000]:
        plt.annotate(xy=[abs(p['countPaid']-300),p['countUnpaid']+50],s=p['provId'])
    plt.title('Paid vs Unpaid Per Provider')
    plt.xlabel('# Paid Claims')
    plt.ylabel('# Unplaid Claims')

    fig.legend(loc="center right")
    # ax.grid(True)
    plt.subplots_adjust(right=0.78)
plt.show()

# %%

percentUnpaid = sum([u['countUnpaid'] for u in plotData])/sum([t['countTotal'] for t in plotData])
percentUnpaid
# %%
def toFloat(obj, col):
    for i,x in enumerate(obj):
        
        try:
            float(x[col]) 
        except:
            print(i, x[col])
    
    return [float(i) for i in [x[col] for x in obj]]
 



# %%
jCodes[:,claims_lables.index('Provider.Payment.Amount')] = toFloat(jCodes, claims_lables.index('Provider.Payment.Amount'))
jData = {v: [val for val in jCodes[:,k]] for k,v in enumerate(claims_lables)}
jData['Prov.Paid'] = [int(float(x)>0) for x in jData['Provider.Payment.Amount']]
jData['Sub.Paid'] = [int(float(x)>0) for x in jData['Subscriber.Payment.Amount']]
# %%

def encoder(obj):
    res = {}
    labeler = LabelEncoder()
    onehot = OneHotEncoder(categories = 'auto', sparse = False)
    for j in obj:
        if (j == 'Prov.Paid' or
            j == 'Sub.Paid' or
            j == "V1" or 
            j == "Claim.Number" or
            j == "Claim.Line.Number" or
            j == "Member.ID" or 
            j == "Claim.Charge.Amount" or
            j == 'Subscriber.Payment.Amount' or
            j == 'Provider.Payment.Amount' or
            j == 'Subscriber.Payed' or
            j == 'Claim.Subscriber.Type' or
            j == 'Subgroup.Index' or
            j == 'Denial.Reason.Code' or
            j == "Place.Of.Service.Code"):
            pass
        else:
            inted = labeler.fit_transform(obj[j])
            oneEnc = onehot.fit_transform(inted.reshape(len(inted), 1))
            res[j] = oneEnc
    res['Claim.Charge.Amount'] = np.array(list(map(float,obj['Claim.Charge.Amount']))).reshape(len(obj['Claim.Charge.Amount']),1)
    res['Subscriber.Index'] = np.array(list(map(float,obj['Subscriber.Index']))).reshape(len(obj['Subscriber.Index']),1)
    #res['Provider.Payment.Amount'] = np.array(list(map(float,obj['Provider.Payment.Amount']))).reshape(len(obj['Claim.Charge.Amount']),1)
    return(res)

# %%
x = encoder(jData)
X = np.hstack(list(x.values()))
Y = np.array(jData['Prov.Paid'])


# %%

def part_one(grids):
    return GridSearch(grids)

def part_three():
    n_folds = 5
    n_splits, random_state = 15, 101

    

    return dict(X=X, y=Y, n_splits=n_splits, random_state=random_state)

def part_four(gs, data):
    try:
        
        gs = loadGrid(output_path+'pt3.pickle')
    except:
        print('*'*20+'\nRUNNING MODELS\n'+'*'*20)
        gs.run(**data)

    gs.vizualizeAll(saveFigsPath=output_path)
    return gs


def part_five(gs):
    gs.save(path=output_path+'pt3.pickle')
    print('GridSearch Object saved at :\n{}'.format(output_path+'pt3.pickle'))
    # gs.pprint(outputPath=output_path+'pt5.txt')


def part_six_seven(gs):
    gs.pprint(outputPath=output_path+'pt3-7.txt')


# %%
def main():
    import inspect
    logReg_Grid1 = Grid(
        logReg(),
        grid_opts=[{'C': [0.1, 1, 10],
                    'max_iter': [100, 500, 1000],
                    'penalty': ['l2']}])

    # rf_Grid = Grid(
    #     rf(),
    #     grid_opts=[{'n_estimators': [10, 100],
    #                 'max_features': ['auto'],
    #                 'max_depth': [1, 5],
    #                 'criterion':['gini', 'entropy']}])

    # svm_Grid = Grid(
    #     svm(),
    #     grid_opts=[{
    #         'C': [0.01, 0.1, 1, 10],
    #         'kernel': ['linear', 'poly']
    #     }, {
    #         'C': [0.01, 0.1, 1, 10],
    #         'gamma': [ 0.001, 0.01, 1],
    #         'kernel': ['rbf']
    #     }])

    # logReg_Grid = Grid(
    #     logReg(),
    #     grid_opts=[{'C': [0.1, 1, 10],
    #                 'max_iter': [100, 500, 1000],
    #                 'penalty': ['l2']}])

    rf_Grid = Grid(
        rf(),
        grid_opts=[{'n_estimators': [100, 200],
                    'max_features': ['auto'],
                    'max_depth': [5,10, 20],
                    'criterion':['gini']}])

    knn_Grid = Grid(
        KNN(),
        grid_opts=[{
            'n_neighbors': [3,5]
        }])

    gnb_Grid = Grid(
        GaussianNB(),
        grid_opts=[{'priors':[None]}])

    logReg_Grid = Grid(
        logReg(),
        grid_opts=[{'C': [0.1, 1, 10],
                    'max_iter': [ 500, 1000],
                    'penalty': ['l2']}])

    
    grids2 = {
        'rf': rf_Grid,
        'gausianNB': gnb_Grid,
        'logReg': logReg_Grid
    }
    resp = {}
    parts = dict(
        pt2=(part_one, grids2),
        pt3=(part_three, False),
        pt4=(part_four, ['pt2', 'pt3']),
        pt5=(part_five, 'pt4'),
        pt67=(part_six_seven, 'pt4')
    )

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


# %%
if __name__ == "__main__":
    main()

# %%
