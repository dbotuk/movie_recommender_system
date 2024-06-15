import pandas as pd
import random
from src.utils import RatingMatrix


class MeanRatingRecommender:
    def __init__(self):
        self.recommendations = pd.DataFrame()

    def train(self, train_ratings):
        """
        Train a Mean Rating Recommender model
        """
        mean_ratings = train_ratings.get_rating_matrix().mean(axis=1).reset_index()

        mean_ratings.columns = ['MovieID', 'Rating']
        mean_ratings.set_index('MovieID', inplace=True)

        self.recommendations = mean_ratings.T

    def predict(self, X_test):
        """
        Provide a Mean Rating Recommendations
        """
        repeated_rows = self.recommendations.loc[self.recommendations.index.repeat(len(X_test.get_users()))].reset_index(drop=True)

        predictions = pd.DataFrame(repeated_rows)
        predictions['UserID'] = X_test.get_users()
        predictions.set_index('UserID', inplace=True)
        predictions = predictions.T
        predictions = predictions.loc[list(set(X_test.get_movies()) & set(predictions.index))]
        new_movies = list(set(X_test.get_movies()) - set(predictions.index))
        for movie in new_movies:
            predictions.loc[movie] = [random.randint(1, 6) for user in X_test.get_users()]

        predictions = predictions.round() * X_test.get_rating_matrix()

        return RatingMatrix(predictions)
