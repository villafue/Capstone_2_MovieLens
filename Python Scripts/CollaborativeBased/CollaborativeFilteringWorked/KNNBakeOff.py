# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:11:13 2018

@author: Frank
"""

from MovieLens import MovieLens
from surprise import KNNBasic
from surprise import KNNWithZScore
from surprise import KNNWithMeans
from surprise import KNNBaseline
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

# User-based KNN - cosine
UserKNNcosine = KNNBasic(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNNcosine, "User KNN cosine")

# User-based KNN - msd
UserKNNmsd = KNNBasic(sim_options = {'name': 'msd', 'user_based': True})
evaluator.AddAlgorithm(UserKNNmsd, "User KNN msd")

# User-based KNN - pearson
UserKNNpearson = KNNBasic(sim_options = {'name': 'pearson', 'user_based': True})
evaluator.AddAlgorithm(UserKNNpearson, "User KNN pearson")

# User-based KNN - pearson_baseline
UserKNNpb = KNNBasic(sim_options = {'name': 'pearson_baseline', 'user_based': True})
evaluator.AddAlgorithm(UserKNNpb, "User KNN pearson_basline")

# User-based KNNZScore 
UserKNNzscore = KNNWithZScore(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNNzscore, "User KNNZScore")

# User-based KNNMeans
UserKNNmeans = KNNWithMeans(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNNmeans, "User KNNMeans")

# User-based KNNBaseline
UserKNNbline = KNNBaseline(sim_options = {'name': 'cosine', 'user_based': True})
evaluator.AddAlgorithm(UserKNNbline, "User KNNZBaseline")

# Item-based KNN - cosine
ItemKNNcosine = KNNBasic(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNNcosine, "Item KNN cosine")

# Item-based KNN - msd
ItemKNNmsd = KNNBasic(sim_options = {'name': 'msd', 'user_based': False})
evaluator.AddAlgorithm(ItemKNNmsd, "Item KNN msd")

# Item-based KNN - pearson
ItemKNNpearson = KNNBasic(sim_options = {'name': 'pearson', 'user_based': False})
evaluator.AddAlgorithm(ItemKNNpearson, "Item KNN pearson")

# Item-based KNN - pearson_baseline
ItemKNNpb = KNNBasic(sim_options = {'name': 'pearson_baseline', 'user_based': False})
evaluator.AddAlgorithm(ItemKNNpb, "Item KNN pearson_basline")

# Item-based KNNZScore 
ItemKNNzscore = KNNWithZScore(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNNzscore, "Item KNNZScore")

# Item-based KNNMeans
ItemKNNmeans = KNNWithMeans(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNNmeans, "Item KNNMeans")

# Item-based KNNBaseline
ItemKNNbline = KNNBaseline(sim_options = {'name': 'cosine', 'user_based': False})
evaluator.AddAlgorithm(ItemKNNbline, "Item KNNZBaseline")


# Just make random recommendations
Random = NormalPredictor()
evaluator.AddAlgorithm(Random, "Random")

# Fight!
evaluator.Evaluate(True)

evaluator.SampleTopNRecs(ml)
