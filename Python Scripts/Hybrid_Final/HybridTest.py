# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from ContentKNNAlgorithm import ContentKNNAlgorithm
from HybridAlgorithm import HybridAlgorithm
from surprise import SVD, SVDpp
from surprise import NormalPredictor
from Evaluator import Evaluator

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

# Construct an Evaluator to, you know, evaluate them
evaluator = Evaluator(evaluationData, rankings)

# RBM
RBM = RBMAlgorithm(epochs=40, hiddenDim=40, learningRate=0.2)

# SVD++ Untuned
SVDPlus = SVDpp()

# SVD Tuned
SVDTuned = SVD(n_epochs=11, lr_all=0.034, n_factors=90, reg_all=0.06)

Random = NormalPredictor()

#Content
#ContentKNN = ContentKNNAlgorithm()

#Combine them
Hybrid = HybridAlgorithm([RBM, SVDPlus, SVDTuned, Random], [0.3, 0.3, 0.3, 0.1])

evaluator.AddAlgorithm(RBM, "RBM")
evaluator.AddAlgorithm(SVDPlus, "SVD++Untuned")
evaluator.AddAlgorithm(SVDTuned, "SVD Tuned")
evaluator.AddAlgorithm(Random, "Random")
#evaluator.AddAlgorithm(ContentKNN, "ContentKNN")
evaluator.AddAlgorithm(Hybrid, "Hybrid")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
