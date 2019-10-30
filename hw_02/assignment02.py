#%%

import os
import sys

# from numpy import genfromtxt
import numpy as np
import re
import csv
from pprint import pprint



#%%
path = os.path.dirname(os.path.abspath(__file__))

# %%
with open(path+'/claim.sample.csv') as file:
    reader = csv.reader(file)
    l = list(map(list, reader))
    
pprint(l[0])
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
with plt.style.context('fivethirtyeight'):
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
    plt.subplots_adjust(right=0.82)
plt.show()

# %%

percentUnpaid = sum([u['countUnpaid'] for u in plotData])/sum([t['countTotal'] for t in plotData])

# %%
jCodes[claims_lables.index('Provider.Payment.Amount')] = float(jCodes[claims_lables.index('Provider.Payment.Amount')])

# %%
