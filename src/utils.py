def train_test_split(ratings, base_col, test_size=0.25):
    """
    Split ratings into train and test sets based on base_col
    """
    cutoff = ratings[base_col].quantile(1 - test_size)

    train = RatingMatrix(ratings[ratings[base_col] <= cutoff].pivot(index='MovieID', columns='UserID', values='Rating'))
    test = RatingMatrix(ratings[ratings[base_col] > cutoff].pivot(index='MovieID', columns='UserID', values='Rating'))

    return train, test


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