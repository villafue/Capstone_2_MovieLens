![banner](https://raw.githubusercontent.com/villafue/Capstone_2_MovieLens/7aa6f7dfc758b781b43f2822c67bd9139d5e95f9/Pictures/README/KNOW%20WHAT%20THEY%20LIKE%20BB.svg)
# Building Recommender Systems with Machine Learning and AI

*Recommendation systems are everywhere, and in today's digital world, it's hard not to be apart of a system's algorithm. Whether it's my purchases on Amazon, what I watch on Youtube, or even my personal playlists on Spotify, my information is used to influence what I (and others) consume next. In fact, in this June, 2020 [article](https://medium.com/swlh/we-know-what-you-like-perks-of-recommendation-systems-in-business-5f227bb6d09) by Miquido, they say that 35% of Amazon's revenue comes from recommendations and 75% of what users watch comes from movie recommendations. Furthermore, the Netflix recommendation system saves them over $1 billion each year! This is not an insignificant amount. It makes sense why businesses incorporate recommender systems as part of their business model.*

*Unfortunately, most personal projects building recommenders evaluate the efficacy of their system using only an accuracy metric such as the [RMSE](https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/), and in the real world, people couldn't care less if an algorithm can predict how many stars he or she would give a movie. All they care about is if they'll love the content that's shown them. This is why I chose to build my recommender systems using [Frank Kane's](https://www.linkedin.com/in/fkane/?trk=lil_course) core framework from his [Building Recommender Systems with Machine Learning and AI](https://www.linkedin.com/learning/building-recommender-systems-with-machine-learning-and-ai/install-anaconda-review-course-materials-and-create-movie-recommendations?u=36492188) course. He uses a holistic and business-driven approach to building and evaluating recommendation systems. I truly believe this approach will make customers happy and ultimately increase revenue.*  

[[Full Notebook]](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/MovieLens.ipynb)

## 1. Data

The [MovieLens](https://en.wikipedia.org/wiki/MovieLens) dataset is very popular in the recommender community. It was created in 1997 by [GroupLens Research](https://grouplens.org/), and the specific dataset used for this project has about 100,000 ratings for 9,000 movies by 600 users. For more information, please click the links below:

> * [MovieLens Website](https://movielens.org/)

> * [MovieLens Official Datasets](https://grouplens.org/datasets/movielens/)

> * [Dataset used in this Project](https://github.com/villafue/Capstone_2_MovieLens/tree/main/Data)

## 2. Framework

[[Framework Notebook]](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/4_Framework.ipynb).

Built Recommenders using Frank's core framework of five modules. His scripts quickly automated the raw data, pre-processed them for modeling, and made it easy to quickly build, tune, and evaluate the different algorithms. 

## 3. Method

There are the types of recommenders used in this project:

1. **Content-based Recommenders:** Content-based systems recommends items based on the attributes of those items themselves, instead of trying to use aggregate user behavior data.

> * [Content-based Recommenders Notebook](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/5_Content_Based_Recommenders.ipynb)

2. **Collaborative-based Recommender:** Collaborative Based Recommenders leverages the behavior or others to inform what a user might enjoy. At a very high level, it means finding other people like him/her and recommending stuff they liked. Or it might mean finding other things similar to the things that he/she likes. Either way, the idea is taking cues from people similar to a specified user and recommending stuff based on the things they like that this user has not seen yet. It's recommending stuff based on other people's collaborative behavior.

> * [Collaborative-based Recommenders Notebook](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/6_Collaborative_Based_Recommenders.ipynb#collaborative)

3. **Matrix Factorization Methods:** Instead of trying to find items or users that are similar to each other, data science and machine learning techniques are applied to extract predictions from the ratings data. The approach is to train models with user-ratings data, and use those models to predict the ratings of new movies by the users.

> * [Matrix Factorizatioin Methods Notebook](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/7_Matrix_Factorization_Methods.ipynb#matrix)

4. **Deep Learning:** Deep learning opens up entirely new approaches to making recommendations that are worth exploring, and allows users to take advantage of all the rapid advances in the field of artificial intelligence.

> * [Deep Learning Recommenders Notebook](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/8_Deep_Learning.ipynb#deep_learning)

5. **Hybrid Recommenders:** In the real world, there’s no need to choose a single algorithm for your recommender system. Each algorithm has its own strengths and weaknesses and combining many algorithms together could make the sum better than its parts.

> * [Hybrid Recommenders Notebook](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/9_Hybrid.ipynb#hybrid)

## 3. Data Preparation 

[[Data Preparation Section]](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#data_preparation)

In a regression problem, the task is to predict the dependent variable given a set of independent features. The goal is measure how closely the predictions match the actual values. To prepare my data, I had to put both train and test sets together, remove outliers, and impute the missing values. 

## 4. EDA

[[EDA Section]](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#exploratory_data_analysis)

* I went through each of the 80 features and analyzed their pattern against the price of the homes.

![](https://raw.githubusercontent.com/villafue/Capstone_1-_Predict_House_Prices/master/Pictures/EDA_BsmtQual.png)

## 5. Pre-Processing

[[Pre-Processing]](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#target_variable)

* I transformed the dependent variable into something that resembles a more normal distribution, as well as corrected skewness for the indendent features. By transforming the features, it helped in prediction especially for the linear-based models. 

![](https://raw.githubusercontent.com/villafue/Capstone_1-_Predict_House_Prices/master/Pictures/SalePrice%20Transformed.png)

## 6. Modeling

[[Modeling Section]](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#modeling)

I chose to use Python's [scikit-learn library](https://scikit-learn.org/stable/) for my regression problem. I prediced the price of homes using a mixture of linear-based and tree-based algorithms against a baseline model. Once the models were optimized to my data, I then used them as input for my meta-stacking algorithm.

![](https://raw.githubusercontent.com/villafue/Capstone_1-_Predict_House_Prices/master/Pictures/Final%20Table.png)

>***NOTE:** I choose RMSE as the accuracy metric over mean absolute error(MAE) because the errors are squared before they are averaged which gives the RMSE a higher weight to large errors. Thus, the RMSE is useful when large errors are undesirable. The smaller the RMSE, the more accurate the prediction because the RMSE takes the square root of the residual errors of the line of best fit. Furthermore, it is also the chosen metric for how my predictions were to be scored.*

**WINNER: [Stacking Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)**

This algorithm inputs models as base predictors, and by default, uses a version of the [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV) algorithm as the final predictor. The optimized version used both my models and the original training set as input for prediction.  

## 7. Predictions

[[Project Report]](https://github.com/villafue/Capstone_1-_Predict_House_Prices/blob/master/Final/Capstone%201%20Final%20Report%203Feb21.pdf)

Upon submission, I had a final RMSE score of 0.12676 which means that, on average, my predicted house prices were $0.13 off for every $1.00. I also included three business recommendations, for the AREC CEO, on how to prepare her company to use this data.

>***NOTE:** The recommendations are under the "Recommendations" section in the "Report" link above.*

## 8. Future Improvements

[[Project Report]](https://github.com/villafue/Capstone_1-_Predict_House_Prices/blob/master/Final/Capstone%201%20Final%20Report%203Feb21.pdf)

* Compare Time Series Analysis with inflation data. There seemed to be a positive correlation of the price of homes against the year it was built. However, the data was not adjusted for inflation so that is something for further exploration.

* Experiment with Bayesian Hyperparameter Tuning. GridSearch is computationally expensive and RandomSearch is a game of chance. Bayesian Optimization is essentially a “smart” RandomSearch that can identify areas that can improve the predictive accuracy.   

* Auto Modeling and Tuning. In [Section 6.4](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#TPOT), I showed my attempt at using the [TPOT](http://epistasislab.github.io/tpot/) automated machine learning tool. The meta-model it chose was very convoluted and had its own API. I would like to explore this more and see how well it ultimately scores.

>***NOTE:** For the full list of future improvements, please go to the "Areas for Further Exploration" section in the "Report" link above.*

## 9. Credits

This project was a huge labor of love and could not have been accomplished without help and guidance. First, Thanks to Pedro, Serigne, Bsivavenu, Jesuscristo, Pavan, and Arun for their amazing notebooks! I drew a lot of inferences, strategies, and insights from them. Links to their work can be seen under the [6.3 - References](https://colab.research.google.com/github/villafue/Capstone_1-_Predict_House_Prices/blob/master/House_Price.ipynb#references) of my notebook. Also, thank you to Kenneth Gil Pascual for his mentoring and guidance has been absolutely invaluable to the quality and completion of this project.




