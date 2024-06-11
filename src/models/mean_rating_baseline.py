import pandas as pd
from src.utils import RatingMatrix


class MeanRatingRecommender:
    def __init__(self):
        self.recommendations = pd.DataFrame()

    def train(self, train_ratings):
        """
        Train a Mean Rating Recommender model
        """
        mean_ratings = pd.DataFrame()
        mean_ratings['Rating'] = train_ratings.get_rating_matrix().mean(axis=1)

        self.recommendations = mean_ratings

    def predict(self, users):
        """
        Provide a Mean Rating Recommendations
        """
        temp_rec = self.recommendations.T
        repeated_rows = temp_rec.loc[temp_rec.index.repeat(len(users))].reset_index(drop=True)

        predictions = pd.DataFrame(repeated_rows)
        predictions['UserID'] = users
        predictions.set_index('UserID', inplace=True)

        return RatingMatrix(predictions.T)
