import pandas as pd
import numpy as np
from src.utils import RatingMatrix


class ALSRecommender:
    def __init__(self):
        self.ratings = pd.DataFrame()
        self.users = pd.DataFrame()
        self.movies = pd.DataFrame()
        self.users_matrix = pd.DataFrame()
        self.movies_matrix = pd.DataFrame()

    def train(self, train_ratings, num_factors, lambda_=0.01, max_steps=100, exp=1e-5):
        """
        Train an ALS model
        """
        self.ratings = train_ratings.get_rating_matrix()
        self.ratings.fillna(0, inplace=True)
        self.ratings = self.ratings.to_numpy()
        self.users = train_ratings.get_users()
        self.movies = train_ratings.get_movies()
        num_users, num_movies = self.ratings.shape
        P = np.random.rand(num_users, num_factors)
        Q = np.random.rand(num_movies, num_factors)
        mask = (self.ratings > 0).astype(int)
        prev_error = 1
        for step in range(max_steps):
            for u in range(num_users):
                Q_T_Q = Q.T @ Q
                lambda_I = lambda_ * np.eye(num_factors)
                P[u, :] = np.linalg.solve(Q_T_Q + lambda_I, self.ratings[u, :] @ Q)

            for i in range(num_movies):
                P_T_P = P.T @ P
                lambda_I = lambda_ * np.eye(num_factors)
                Q[i, :] = np.linalg.solve(P_T_P + lambda_I, self.ratings[:, i].T @ P)

            reconstruction = P @ Q.T
            error = np.sqrt(np.mean((mask * (self.ratings - reconstruction.round())) ** 2))

            if abs(error - prev_error) < exp:
                break
            prev_error = error

        self.users_matrix = P
        self.movies_matrix = Q.T


    def predict(self, X_test):
        """
        Provide an ALS Recommendations
        """
        temp_users = [user for user in X_test.get_users() if user in self.users]
        temp_movies = [movie for movie in X_test.get_movies() if movie in self.movies]
        predictions = pd.DataFrame(np.zeros((len(temp_movies), len(temp_users))), index=temp_movies, columns=temp_users)
        predictions.index.names = ['MovieID']
        predictions.columns.names = ['UserID']

        predicted_ratings = np.dot(self.users_matrix, self.movies_matrix)

        predicted_ratings_df = pd.DataFrame(predicted_ratings, index=self.movies, columns=self.users)

        predicted_ratings_df = predicted_ratings_df.loc[temp_movies][temp_users]
        predicted_ratings_df = predicted_ratings_df.round() * X_test.get_rating_matrix().loc[temp_movies][temp_users]

        return RatingMatrix(predicted_ratings_df)
