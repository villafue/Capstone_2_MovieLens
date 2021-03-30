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

## 2. Evaluation Metrics

[[Framework Notebook]](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/4_Framework.ipynb)

1. There are eight quantitative metrics used to evaluate each recommender system: Two for accuracy, three that are user-focused, and three that evaluate the over-all system itself:

```
Legend:

RMSE:      Root Mean Squared Error. Lower values mean better accuracy.
MAE:       Mean Absolute Error. Lower values mean better accuracy.
HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.
cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.
ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better.
Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.
Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations
           for a given user. Higher means more diverse.
Novelty:   Average popularity rank of recommended items. Higher means more novel.
```

2. There is also one qualitative metric used to round out the evaluation of a model, and it prints out the top-10 movie recommendations for a specified user: 

```
Using recommender  ContentKNN

We recommend:
Ant-Man and the Wasp (2018) 4.7450497998793395
The Darkest Minds (2018) 4.7450497998793395
Annihilation (2018) 4.7450497998793395
Game Night (2018) 4.7450497998793395
Tomb Raider (2018) 4.7450497998793395
Alpha (2018) 4.7450497998793395
Solo: A Star Wars Story (2018) 4.7450497998793395
Fred Armisen: Standup for Drummers (2018) 4.7450497998793395
Tom Segura: Disgraceful (2018) 4.7450497998793395
When We First Met (2018) 4.7450497998793395
```

I chose "User 25" and more information can be found in the [[EDA Notebook]](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/3_Exploratory_Data_Analysis.ipynb).

## 3. Models

There are the types of recommenders used in this project:

1. **Content-based Recommenders:** Content-based systems recommends items based on the attributes of those items themselves, instead of trying to use aggregate user behavior data.

2. **Collaborative-based Recommender:** Collaborative Based Recommenders leverages the behavior or others to inform what a user might enjoy. At a very high level, it means finding other people like him/her and recommending stuff they liked. Or it might mean finding other things similar to the things that he/she likes. Either way, the idea is taking cues from people similar to a specified user and recommending stuff based on the things they like that this user has not seen yet. It's recommending stuff based on other people's collaborative behavior.

3. **Matrix Factorization Methods:** Instead of trying to find items or users that are similar to each other, data science and machine learning techniques are applied to extract predictions from the ratings data. The approach is to train models with user-ratings data, and use those models to predict the ratings of new movies by the users.

4. **Deep Learning:** Deep learning opens up entirely new approaches to making recommendations that are worth exploring, and allows users to take advantage of all the rapid advances in the field of artificial intelligence.

5. **Hybrid Recommenders:** In the real world, there’s no need to choose a single algorithm for your recommender system. Each algorithm has its own strengths and weaknesses and combining many algorithms together could make the sum better than its parts.

The Notebooks for each method can be found here:

> * [Content-based Recommenders Notebook](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/5_Content_Based_Recommenders.ipynb)
> * [Collaborative-based Recommenders Notebook](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/6_Collaborative_Based_Recommenders.ipynb#collaborative)
> * [Matrix Factorizatioin Methods Notebook](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/7_Matrix_Factorization_Methods.ipynb#matrix)
> * [Deep Learning Recommenders Notebook](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/8_Deep_Learning.ipynb#deep_learning)
> * [Hybrid Recommenders Notebook](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/Notebook/9_Hybrid.ipynb#hybrid)

## 4. Conclusion

[[Conclusion Section]](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/MovieLens.ipynb#conclusion)

1. The following is a table of the best recommenders from each section:

```
Algorithm         RMSE       MAE        HR         cHR        ARHR       Coverage   Diversity  Novelty   
RBM - Tuned       1.2797     1.0856     0.0230     0.0230     0.0089     0.0000     0.0574     965.0569
ContentKNN        0.9336     0.7224     0.0049     0.0049     0.0012     0.5492     0.3442     2981.5376
Item KNNZBaseline 0.8931     0.6866     0.0246     0.0246     0.0094     0.9525     0.5054     4512.4656
SVD - Tuned       0.8665     0.6626     0.0262     0.0262     0.0076     0.9672     0.2945     2030.2127
Hybrid - 9.4.07   0.8577     0.6594     0.0328     0.0328     0.0137     0.9295     0.1716     1372.1110
```

2. Out of all the models in my project, if I had to choose one, it would be:

```
Algorithm         RMSE       MAE        HR         cHR        ARHR       Coverage   Diversity  Novelty 
Hybrid - 9.4.07   0.8577     0.6594     0.0328     0.0328     0.0137     0.9295     0.1716     1372.1110

-----

Using recommender  Hybrid - 9.4.07

We recommend:
Usual Suspects, The (1995) 5
Pulp Fiction (1994) 5
Star Wars: Episode V - The Empire Strikes Back (1980) 5
Princess Bride, The (1987) 5
Fight Club (1999) 5
Shawshank Redemption, The (1994) 5
Departed, The (2006) 5
Philadelphia Story, The (1940) 5
Life Is Beautiful (La Vita Ã¨ bella) (1997) 5
In the Mood For Love (Fa yeung nin wa) (2000) 5
```

I chose it because it had the best accuracy (lowest RMSE/MAE), the highest hit rates, and a decent Novelty score. Furthermore, its top-10 recommendations were something I would be proud to recommend for A/B testing.

3. However, in the real-world, there is not just one "Top-N" list of recommendations but multiple on an app or website. In [Section 11.1](https://colab.research.google.com/github/villafue/Capstone_2_MovieLens/blob/main/MovieLens.ipynb#afterward), I recommend different algorithms for specific use-cases. More information can be found in the [[Project Report]](https://github.com/villafue/Capstone_2_MovieLens/blob/main/Final/MovieLens%20Capstone%202%20Project%20Report.pdf).  

## 5. Future Improvements

[[Project Report]](https://github.com/villafue/Capstone_2_MovieLens/blob/main/Final/MovieLens%20Capstone%202%20Project%20Report.pdf)

* Frank said one of the biggest issues was data sparsity, and admitted that 100k ratings is not enough information. Because of this, some models did not perform as well or as optimal as possible. One avenue I would like to take is to use the full 27 million rating dataset from [GroupLens.org](https://grouplens.org/datasets/movielens/)

* Exploring the full 27 million dataset would probably require me to use a platform like Apache Spark to spread all the data among clusters. Using this platform would be a worthy endeavor in and of itself.

* Other areas of improvement stems from technical issues. I could only use RandomizedSearchCV with one of my algorithms. With the exception of SVD, I'm not confident in my hyperparameter tuning as it took all day to run and I could only use GridSearchCV. Also, Frank admitted the Hybrid algorithm was rather simple. I had to input the weights of each algorithm and that was how my Hybrid recommender was created. I would like to one day partner with a programmer or do it myself where the weights are learned for each Hybrid model.

>***NOTE:** For the full list of future improvements, please go to the "Areas for Further Exploration" section in the "Report" link above.*

## 6. Credits

This project would not have come to pass without the help or knowledge from the following people:

1.   Kenneth Gil-Pasquel, my Springboard Data Science Advisor, was essential in the quality and timely completion of this project.  
2.   [Frank Kane](https://www.linkedin.com/in/fkane/?trk=lil_course) and his [Building Recommender Systems with Machine Learning and AI](https://sundog-education.com/course/building-recommender-systems-with-machine-learning-and-ai/) is the framework my recommender systems were built upon. Furthermore, I loved how he explained recommendation systems from a practical and business standpoint, rather than just a scientific one (i.e. just trying to achieve the lowest RMSE possible).
3. Tamas Bakos and his [Movie Recommendation Algorithm](https://www.kaggle.com/bakostamas/movie-recommendation-algorithm) notebook on Kaggle. It helped while I was doing my EDA.
4. Jagannath Neupane and his [Analysis of MovieLens Dataset (Beginner's Analysis)](https://www.kaggle.com/jneupane12/analysis-of-movielens-dataset-beginner-sanalysis) notebook on Kaggle. His EDA was very helpful and I used his code to create the WordCount visual.



