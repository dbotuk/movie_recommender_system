import numpy as np


def train_test_split(ratings, base_col, test_size=0.25):
    """
    Split ratings into train and test sets based on base_col
    """
    cutoff = ratings[base_col].quantile(1 - test_size)

    train = ratings[ratings[base_col] <= cutoff]
    test = ratings[ratings[base_col] > cutoff]

    return train, test


def to_user_movie_matrix(ratings):
    """
    Converts ratings dataframe into custom RatingMatrix with user ids in columns and movie ids in rows
    """
    return RatingMatrix(ratings.pivot(index='MovieID', columns='UserID', values='Rating'))


def make_binary_matrix(original_matrix):
    """
    Returns the matrix with elements equal to 1 if present else 0
    """
    return RatingMatrix((~np.isnan(original_matrix)).astype(int))


class RatingMatrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def get_user_ratings(self, user_id):
        return self.matrix[user_id]

    def get_movie_ratings(self, movie_id):
        return self.matrix.loc[movie_id]

    def get_rating(self, user_id, movie_id):
        return self.matrix.loc[movie_id][user_id]

    def get_rating_matrix(self):
        return self.matrix

    def get_movies(self):
        return self.matrix.index

    def get_users(self):
        return self.matrix.columns
