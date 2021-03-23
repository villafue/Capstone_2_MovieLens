# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 23:32:16 2021

@author: Maribel
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
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
param_grid = {'hiddenDim': [10, 20, 30, 40, 50, 55, 60, 70, 80, 90, 100], 
              'learningRate': [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
                               0.06, 0.07, 0.08, 0.09, 0.10]}
rs = RandomizedSearchCV(RBMAlgorithm, param_distributions = param_grid, n_jobs=-1,
                        measures=['rmse', 'mae'], cv=5, n_iter=50)

rs.fit(evaluationData)

# best RMSE score
print("Best RMSE score attained: ", rs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(rs.best_params['rmse'])

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

params = rs.best_params['rmse']
RBMtuned = RBMAlgorithm(hiddenDim = params['hiddenDim'], learningRate = params['learningRate'])
evaluator.AddAlgorithm(RBMtuned, "RBM - Tuned")

RBMUntuned = RBMAlgorithm()
evaluator.AddAlgorithm(RBMUntuned, "RBM - Untuned")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
