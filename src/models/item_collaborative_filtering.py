import numpy as np
import pandas as pd
from src.utils import RatingMatrix, make_binary_matrix


class ItemCollaborativeFiltering:
    def __init__(self):
        self.item_similarity = None
        self.train_matrix = None
        self.binary_matrix = None

    def fit(self, train_ratings):
        """
        Fit the model by calculating the item-item similarity matrix.
        """
        self.train_matrix = train_ratings.get_rating_matrix()
        self.binary_matrix = make_binary_matrix(self.train_matrix).get_rating_matrix()
        self.item_similarity = self._calculate_item_similarity(self.binary_matrix)


    def _calculate_item_similarity(self, matrix):
        """
        Calculate cosine similarity between items.
        """
        similarity = np.dot(matrix, matrix.T)
        norms = np.array([np.sqrt(np.diagonal(similarity))])
        return similarity / norms / norms.T

    def predict(self, user_id, movie_id):
        """
        Predict the rating a user would give to a movie.
        """
        if user_id not in self.train_matrix.columns or movie_id not in self.train_matrix.index:
            raise ValueError(f"User {user_id} or Movie {movie_id} not found in the training data.")

        movie_idx = self.train_matrix.index.get_loc(movie_id)
        movie_sim = self.item_similarity[movie_idx]
        user_ratings = self.train_matrix[user_id]

        if user_ratings.isna().all():
            return np.nan

        mean_movie_rating = self.train_matrix.loc[movie_id].mean()

        mask = ~user_ratings.isna()
        if mask.sum() == 0:
            return mean_movie_rating  # No neighbors have rated the movie

        neighbor_ratings = self.train_matrix.loc[mask, user_id]
        movie_means = self.train_matrix.loc[mask].mean(axis=1)

        similarity_sum = np.sum(movie_sim[mask])
        weighted_sum = np.sum(movie_sim[mask] * (neighbor_ratings - movie_means))

        prediction = mean_movie_rating + weighted_sum / (similarity_sum + 1e-9)

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
