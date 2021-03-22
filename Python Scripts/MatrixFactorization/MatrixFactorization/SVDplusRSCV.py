# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 23:50:53 2021

@author: Maribel
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 22:35:07 2021

@author: Maribel
"""

from MovieLens import MovieLens
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV
from surprise.model_selection import RandomizedSearchCV

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(29)
random.seed(29)

# Load up common data set for the recommender algorithms
(ml, evaluationData, rankings) = LoadMovieLensData()

print("Searching for best parameters...")
#param_grid = {'n_epochs': [20, 30], 'lr_all': [0.005, 0.010],
#              'n_factors': [50, 100]}
#gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

#gs.fit(evaluationData)

param_grid = {'n_epochs': range(10, 31), 
              'lr_all': np.linspace(0.001, 0.10, 10),
              'n_factors': range(10, 100, 20),
              'reg_all': np.linspace(0.01, 0.1, 10)}

rsplus = RandomizedSearchCV(SVDpp, param_distributions=param_grid, measures=['rmse', 'mae'], 
                        cv=2, random_state=29, n_jobs = -1, n_iter=2)

rsplus.fit(evaluationData)

# best RMSE score
#print("Best RMSE score attained: ", gs.best_score['rmse'])
print("Best RMSE score attained: ", rsplus.best_score['rmse'])

# combination of parameters that gave the best RMSE score
#print(gs.best_params['rmse'])
print(rsplus.best_params['rmse'])

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

#params = gs.best_params['rmse']
#SVDtuned = SVD(n_epochs = params['n_epochs'], lr_all = params['lr_all'], n_factors = params['n_factors'])

params = rsplus.best_params['rmse']
SVDPlusTuned = SVDpp(n_epochs = params['n_epochs'], lr_all = params['lr_all'], 
               n_factors = params['n_factors'], reg_all = params['reg_all'])
evaluator.AddAlgorithm(SVDPlusTuned, "SVD++ - Tuned")

SVDPlusUntuned = SVDpp()
evaluator.AddAlgorithm(SVDPlusUntuned, "SVD++ - Untuned")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
