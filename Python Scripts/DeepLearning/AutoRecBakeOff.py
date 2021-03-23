# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from MovieLens import MovieLens
from AutoRecAlgorithm import AutoRecAlgorithm
from surprise import NormalPredictor
from Evaluator import Evaluator
from surprise.model_selection import GridSearchCV

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
param_grid = {'hiddenDim': [100, 200], 'learningRate': [0.1, 0.2]}
gs = GridSearchCV(AutoRecAlgorithm, param_grid, n_jobs=-1, measures=['rmse', 'mae'], cv=2)

gs.fit(evaluationData)

# best RMSE score
print("Best RMSE score attained: ", gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

params = gs.best_params['rmse']
AutoRectuned = AutoRecAlgorithm(hiddenDim = params['hiddenDim'], learningRate = params['learningRate'])
evaluator.AddAlgorithm(AutoRectuned, "AutoRec - Tuned")

#Autoencoder
AutoRecUntuned = AutoRecAlgorithm()
evaluator.AddAlgorithm(AutoRecUntuned, "AutoRec - Untuned")

# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
