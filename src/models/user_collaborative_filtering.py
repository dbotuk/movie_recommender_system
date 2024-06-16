import numpy as np
import pandas as pd
from src.utils import RatingMatrix


class UserCollaborativeFiltering:
    def __init__(self):
        self.user_similarity = None
        self.train_matrix = None

    def fit(self, train_ratings):
        """
        Fit the model by calculating the user-user similarity matrix.
        """
        self.train_matrix = train_ratings.get_rating_matrix()
        self.user_similarity = self._calculate_user_similarity(self.train_matrix)

    def _calculate_user_similarity(self, matrix):
        """
        Calculate cosine similarity between users.
        """
        matrix_filled = matrix.fillna(0)
        similarity = np.dot(matrix_filled.T, matrix_filled)
        norms = np.array([np.sqrt(np.diagonal(similarity))])
        return similarity / norms / norms.T

    def predict(self, user_id, movie_id):
        """
        Predict the rating a user would give to a movie.
        """
        if user_id not in self.train_matrix.columns or movie_id not in self.train_matrix.index:
            raise ValueError(f"User {user_id} or Movie {movie_id} not found in the training data.")

        # Get the column index for the user
        user_idx = self.train_matrix.columns.get_loc(user_id)

        # Extract the similarities for the user
        user_sim = self.user_similarity[user_idx]

        # Extract the ratings for the movie
        ratings = self.train_matrix.loc[movie_id]

        if ratings.isna().all():
            return np.nan  # No ratings available

        mean_user_rating = self.train_matrix[user_id].mean()

        mask = ~ratings.isna()
        if mask.sum() == 0:
            return mean_user_rating  # No neighbors have rated the movie

        # Align the mean calculation with the other users who have rated this movie
        neighbor_ratings = self.train_matrix.loc[movie_id, mask]
        user_mean = self.train_matrix.loc[:, mask].mean()

        similarity_sum = np.sum(user_sim[mask])
        weighted_sum = np.sum(user_sim[mask] * (neighbor_ratings - user_mean))

        prediction = mean_user_rating + weighted_sum / (similarity_sum + 1e-9)

        # Cap the predicted rating between 1 and 5
        prediction = max(1, min(prediction, 5))

        return prediction

    def recommend(self, user_id, n_recommendations=5):
        """
        Recommend top N movies for a user.
        """
        if user_id not in self.train_matrix.columns:
            raise ValueError(f"User {user_id} not found in the training data.")

        user_ratings = self.train_matrix[user_id]
        all_movies = self.train_matrix.index
        predicted_ratings = []

        for movie_id in all_movies:
            if pd.isna(user_ratings[movie_id]):
                predicted_rating = self.predict(user_id, movie_id)
                if not np.isnan(predicted_rating):
                    predicted_ratings.append((movie_id, predicted_rating))

        predicted_ratings.sort(key=lambda x: x[1], reverse=True)
        return predicted_ratings[:n_recommendations]
