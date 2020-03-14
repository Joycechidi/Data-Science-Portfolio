
# Introduction
This project focuses on creating a recommendation system for users of IBM Watson. There are many data science and machine learning tutorials, datasets and helpful jupyter notebook resources available on the Watson Studio website. 

I used the user-user collaborative filtering method to recommend articles to other users based on similarity of interactions between users. 
User to User recommendations is a good recommendation method because it provides solution or insight to the most frequently interacted articles and frequent users. New users who have no data in the system would be interested to know why a particular article is more popular than others and have a high probability of reading the article.

If the new user interacts with the article, the next step will be to get a review of their sentiments about the article. This way we will be making the first steps in getting the new user to make more interactions.
Rank-based recommendation method was used to provide the top articles based on number of interactions by users.

I also used Matrix Factorization to make article recommendations to the users on the IBM Watson Studion platform. Singular Value Decomposition was used on the User to Article matrix. All the latent features in the data was used because I noticed that as the number of latent features increased, lower error rates were obtained when making predictions for the 1 abd 0 values in the user-article matrix.

To understand the impact on recommendation accuracy, the data was split into training and test sets of data. The underlying questions that were answered from these training were:
1. How many users can we make predictions for in the test set?
2. How many users are we not able to make predictions for because of the cold start problem?
3. How many articles can we make predictions for in the test set?
4. How many articles are we not able to make predictions for because of the cold start problem?

# Libraries Used
1. numpy
2. pandas
3. matplotlib
4. seaborn
5. plotly
6. pickle

# Data Sources
IBM Watson Studio provided by Udacity

# Author
Solution and Code written by Joyce Chidiadi
