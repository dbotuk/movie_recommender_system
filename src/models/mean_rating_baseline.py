import pandas as pd
import numpy as np
import random
from src.utils import RatingMatrix


class MeanRatingRecommender:

    def __init__(self):
        self.ratings = pd.DataFrame()

    def train(self, train_ratings):
        """
        Train a Mean Rating Recommender model
        """
        mean_ratings_per_movie = train_ratings.get_rating_matrix().mean(axis=1).reset_index()

        mean_ratings_per_movie.columns = ['MovieID', 'Rating']
        mean_ratings_per_movie.set_index('MovieID', inplace=True)

        self.ratings = mean_ratings_per_movie.T.round()

    def predict(self, X_test, random_seed=42):
        """
        Provide a Mean Rating Recommendations
        """

        # init of matrix with random ratings for all users from test which are present in training dataset
        # all the movies from X_test, which are not present in training dataset will get random rating
        temp_users = [user for user in X_test.get_users() if user in self.ratings.columns]
        np.random.seed(random_seed)
        random_matrix = np.random.randint(1, 6, size=(len(X_test.get_movies()), len(temp_users)))
        predictions = pd.DataFrame(random_matrix, index=X_test.get_movies(), columns=temp_users)

        repeated_rows = self.ratings.loc[self.ratings.index.repeat(len(X_test.get_users()))].reset_index(drop=True).round()
        repeated_rows['UserID'] = X_test.get_users()
        predictions.update(repeated_rows.T)

        # set all the predicted ratings, which are not present in X_test, to zero
        predictions = predictions * X_test.get_rating_matrix()[temp_users]

        return RatingMatrix(predictions)
