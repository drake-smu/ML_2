
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import metrics
import matplotlib.cm as cm  #https://matplotlib.org/api/cm_api.html
from scipy.stats import rankdata
import warnings
warnings.filterwarnings("ignore")

#%%
#################################
### questions 
#################################

questions = [
    '''
    > Informally describe what a linear combination is  
    > and how it will relate to our resturant matrix.
    ''',
    '''
    > Choose a person and compute(using a linear combination)
    > top restaurant for them. What does each entry in the 
    > resulting vector represent.
    ''',
    '''
    > Next compute a new matrix (M_usr_x_rest i.e. 
    > an user by restaurant) from all people.  
    > What does the a_ij matrix represent?
    ''',
    '''
    > Sum all columns in M_usr_x_rest to get optimal 
    > restaurant for all users. What do the entry’s 
    > represent?
    ''',
    '''
    > Now convert each row in the M_usr_x_rest into a ranking 
    > for each user and call it M_usr_x_rest_rank.   
    > Do the same as above to generate the optimal restaurant 
    > choice.
    ''',
    '''
    > Why is there a difference between the two?  
    > What problem arrives?  
    > What does represent in the real world?
    ''',
    '''
    > How should you preprocess your data to remove 
    > this problem.
    ''',
    '''
    > Find user profiles that are problematic, 
    > explain why?
    ''',
    '''
    > Think of two metrics to compute the disatistifaction 
    > with the group.
    ''',
    '''
    > Should you split in two groups today?
    ''',
    '''
    > Ok. Now you just found out the boss is paying 
    > for the meal. How should you adjust. 
    > Now what is best restaurant?
    ''',
    '''
    > Tommorow you visit another team. You have the 
    > same restaurants and they told you their optimal 
    > ordering for restaurants. Can you find their
    > weight matrix?
    '''
]



#%%
#################################
### PEOPLE 
#################################
people = {'Jane': {'willingness to travel': 0.1596993,
                  'desire for new experience':0.67131344,
                  'cost':0.15006726,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01892,
                  },
          'Bob': {'willingness to travel': 0.63124581,
                  'desire for new experience':0.20269888,
                  'cost':0.01354308,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.15251223,
                  },
          'Mary': {'willingness to travel': 0.49337138 ,
                  'desire for new experience': 0.41879654,
                  'cost': 0.05525843,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.03257365,
                  },
          'Mike': {'willingness to travel': 0.08936756,
                  'desire for new experience': 0.14813813,
                  'cost': 0.43602425,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.32647006,
                  },
          'Alice': {'willingness to travel': 0.05846052,
                  'desire for new experience': 0.6550466,
                  'cost': 0.1020457,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.18444717,
                  },
          'Skip': {'willingness to travel': 0.08534087,
                  'desire for new experience': 0.20286902,
                  'cost': 0.49978215,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.21200796,
                  },
          'Kira': {'willingness to travel': 0.14621567,
                  'desire for new experience': 0.08325185,
                  'cost': 0.59864525,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.17188723,
                  },
          'Moe': {'willingness to travel': 0.05101531,
                  'desire for new experience': 0.03976796,
                  'cost': 0.06372092,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.84549581,
                  },
          'Sara': {'willingness to travel': 0.18780828,
                  'desire for new experience': 0.59094026,
                  'cost': 0.08490399,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.13634747,
                  },
          'Tom': {'willingness to travel': 0.77606127,
                  'desire for new experience': 0.06586204,
                  'cost': 0.14484121,
                  #'indian food':1,
                  #'Mexican food':1,
                  #'hipster points':3,
                  'vegetarian': 0.01323548,
                  }                  
          }

peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people.items():
    row = []
    
    for k2, v2 in v1.items():
        peopleKeys.append(k1+'_'+k2)
        if k1 == lastKey:
            row.append(v2)      
            lastKey = k1
            
        else:
            peopleValues.append(row)
            row.append(v2)   
            lastKey = k1
            

#here are some lists that show column keys and values
# print(peopleKeys,'\n')
# print(peopleValues)

peopleMatrix = np.array(peopleValues)

peopleMatrix.shape

#################################
### RESTAURANTS 
#################################

restaurants  = {'flacos':{'distance' : 2,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                        },
              'Joes':{'distance' : 5,
                        'novelty' : 1,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Poke':{'distance' : 4,
                        'novelty' : 2,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },                      
              'Sush-shi':{'distance' : 4,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },
              'Chick Fillet':{'distance' : 3,
                        'novelty' : 2,
                        'cost': 5,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                      },
              'Mackie Des':{'distance' : 2,
                        'novelty' : 3,
                        'cost': 4,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      },
              'Michaels':{'distance' : 2,
                        'novelty' : 1,
                        'cost': 1,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 5
                      },
              'Amaze':{'distance' : 3,
                        'novelty' : 5,
                        'cost': 2,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 4
                      },
              'Kappa':{'distance' : 5,
                        'novelty' : 1,
                        'cost': 2,
                        #'average rating': 5,
                        #'cuisine': 5,
                        'vegetarian': 3
                      }                  
}
# print(restaurants)

restaurantsKeys, restaurantsValues = [], []

for k1, v1 in restaurants.items():
    for k2, v2 in v1.items():
        restaurantsKeys.append(k1+'_'+k2)
        restaurantsValues.append(v2)

# print(restaurantsKeys)
# print(restaurantsValues)

# len(restaurantsValues)

restaurantsMatrix = np.reshape(restaurantsValues, (9,4))

restaurantsMatrix

restaurantsMatrix.shape

#%%
def printS(section, output):
    pre = 'Pt. {}  '.format(section)+'*'*40+'\n{}\n'.format(questions[section-1])
    print(pre, output, sep='\n', end='\n\n')
# %%
def partTwo(personId, pMatrix, rMatrix):
    _person = pMatrix[personId]
    # print('{} * {}'.format(restaurantsMatrix.shape, _person.T.shape))
    resp = rMatrix.dot(_person.T)
    rank = rankdata(len(resp)-rankdata(resp))
    return dict(zip(rank,restaurants.keys()))

def partThree(pM, rM):
    # print('{} * {}'.format(rM.shape, pM.T.shape))
    return rM.dot(pM.T)

#byScore
def partFour(u_r_m, rNames):
    # print(u_r_m[::-1].shape)
    resp = np.sum(u_r_m.T, axis=0)
    # print(resp.shape)
    rank = rankdata(len(resp)-rankdata(resp))
    # print('\n',rank.shape,resp)
    return dict(zip(rank,rNames))

#byRank
def partFive(u_r_m, rNames):
    temp=u_r_m.T.argsort()
    ranks = np.arange(len(u_r_m.T))[temp.argsort()]+1
    # ranks = rankdata(len(resp)-rankdata(resp))
    sum_rank = np.sum(ranks, axis=0) 
    resp = np.arange(len(sum_rank))[sum_rank.argsort()[::-1]]+1
    # print('\n',sum_rank.shape, sum_rank)
    return dict(zip(resp,rNames))

def partEight(pM, pNames):
    ranges = []
    for i,person in enumerate(pM):
        _r = person.max()-person.min()
        ranges.append((_r))
    # print('Max: {}\nMin: {}\n    range: {}'.format(person.max(),person.min(), _r))

    range_rank = rankdata(len(ranges)-rankdata(ranges))
    return dict(zip(range_rank, pNames))


def partNine(pM, rM):
    pca = PCA(n_components=2)  
    peopleMatrixPcaTransform = pca.fit_transform(pM)
    restaurantsMatrixPcaTransform = pca.fit_transform(rM)    
    print('''
    *  The Davies-Bouldin Index is used to measure better defined 
    clusters. The Davies-Bouldin score is lower when clusters 
    more separated (e.g. better partitioned). Zero is the 
    lowest possible Davies-Bouldin score.

    *  The Calinski-Harabaz Index is used to measure better defined
    clusters. The Calinski-Harabaz score is higher when clusters 
    are dense and well separated.
    ''')
    range_n_clusters = np.arange(2,9)
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(peopleMatrixPcaTransform)
        score1 = metrics.davies_bouldin_score(peopleMatrixPcaTransform, cluster_labels)  
        print("The Davies-Bouldin score for :\n", n_clusters, " clusters is: ", score1)
        score2 = metrics.calinski_harabaz_score(peopleMatrixPcaTransform, cluster_labels)  
        print("The Calinski-Harabaz score for :\n", n_clusters, " clusters is: ", score2)
        print('')

        centroid = clusterer.cluster_centers_
        labels = clusterer.labels_

        print (centroid)
        print(labels)


        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

        #https://matplotlib.org/users/colors.html
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        labelList = ['Jane', 'Bob', 'Mary', 'Mike', 'Alice', 'Skip', 'Kira', 'Moe', 'Sara', 'Tom']

        for i in range(len(peopleMatrixPcaTransform)):
            # print ("coordinate:" , peopleMatrixPcaTransform[i], "label:", labels[i])
            ax.plot(peopleMatrixPcaTransform[i][0],peopleMatrixPcaTransform[i][1],colors[labels[i]],markersize=10)
            #https://matplotlib.org/users/annotations_intro.html
            #https://matplotlib.org/users/text_intro.html
            ax.annotate(labelList[i], (peopleMatrixPcaTransform[i][0],peopleMatrixPcaTransform[i][1]), size=15)
        ax.scatter(centroid[:,0],centroid[:,1], marker = "x", s=150, linewidths = 5, zorder =10)

        plt.show()

def partEleven(pM, rM, rNames):
    _pm = pM
    _pm[:,2] = 0
    res = rM.dot(_pm.T)
    # print(_pm.T,'\n',res.shape,'\n', res)
    byScore = partFour(res, rNames)
    byRank = partFive(res, rNames)
    # print(byScore[1],byRank[1])
    return (byScore, byRank)

# print('Rest shape: {}\nPeople Shape: {}'.format(restaurantsMatrix.shape, peopleMatrix.T.shape))
p2 = partTwo(0,peopleMatrix, restaurantsMatrix)
p3 = partThree(peopleMatrix,restaurantsMatrix)
p4 = partFour(p3.copy(), restaurants.keys())
p5 = partFive(p3.copy(), restaurants.keys())
p8 = partEight(peopleMatrix.copy(), people.keys())
p11a, p11b = partEleven(peopleMatrix.copy(), restaurantsMatrix.copy(), restaurants.keys())

#%%

pr1 = '''\
    A linear combination is an expression of a set of terms that is
    constructed by multipying the terms by some constant and adding the
    results to form a singal scalar Ex:  Vector [1,2,3,4,5]  ;
    scalars/coefficients [a,b,c,d,e] Linear_Combination = a*1 + b*2 + c*3 +
    d*4 + e*5
    '''
pr2 = '''\
    Jane\'s top restaurant is : {}
    Each entry in the resulting vector
    is her score (cost sum) for each
    restaurant.
    '''.format(p2[1])
pr3 = '''\
    M_usr_x_rest :\n{}
    a_ij represents the score (cost) 
    for user_i at restraunt_j
    '''.format(p3)
pr4 = 'The optimal restaurant based on scores\nis {}'.format(p4[1])
pr5 = 'The optimal restaurant based on rankings\nis {}'.format(p5[1])
pr6 = '''\
    The problem is that the results do not matchup.
    This represents the issue with relative rankings.
    When the scores are converted to ordinal ranks,
    information describing the magnitude of preference
    for one choice over another.
    '''
pr7 = '''\
    One preprocessing option is to normalize
    peoples scores, so that each person's 
    score is of similar range while still 
    preserving the magnitude of preference 
    for each restaurant choice.
    '''
pr8 = '''\
    The ones with the most extreme ranges
    (2 most, 2 least) are ({}, {}) and ({}, {}).
    These are the worst because their scores had 
    the greatest and least amount of range. Meaning
    they were the most and least represented when 
    summing all the scores for the restaurants.
    (**Note: this still does not preserve info lost
    when converting to matrix of rankings.**)
    '''.format(p8[1],p8[2],p8[9],p8[10])
pr9 = '''\
    Two metrics to measure disatisfaction would be
    the average distance between the avg. group restaurant
    rating for the restaurants and the individual
    people's rating. The other is the average distance
    between the rankings of the group sums vs the individual
    ranking.
    '''
pr10 = '''\
    Yes, there isn't a clear favorite.
    '''
pr11 = '''\
    If the boss is paying then the importance
    of cost now not relevant. For the peoples
    matrix, we can now assign everyone a zero
    for not important. Before the two answers
    were {} (scores) and {} (ranks). Now they
    are {} (scores) and {} (ranks).
    '''.format(p4[1],p5[1],p11a[1], p11b[1])

pr12 = '''\
    Yes I "could" find their weighted matrix....
    but im not going to let that process ruin
    my lunch so I will pass on it.
    '''

printS(1, pr1)
printS(2, pr2)
printS(3, pr3)
printS(4, pr4)
printS(5, pr5)
printS(6, pr6)
printS(7, pr7)
printS(8,pr8)
printS(9,pr9)
partNine(p3.copy().T, restaurantsMatrix.copy())
printS(10,pr10)
printS(11,pr11)
printS(12,pr12)

# %%
plot_dims = (12,10)
fig, ax = plt.subplots(figsize=plot_dims)
sns.heatmap(ax=ax, data=p3, annot=True)
plt.show()

# %%

