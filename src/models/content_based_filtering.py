import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import RatingMatrix


class ContentBasedFilteringRecommender:
    def __init__(self):
        self.centralized_ratings = pd.DataFrame()
        self.ratings_means = pd.DataFrame()
        self.ratings = pd.DataFrame()
        self.similarity_matrix_df = None
        self.movies = pd.DataFrame()

    def train(self, train_ratings, movies):
        """
        Train a Content-based Filtering model
        """
        self.movies = movies.copy()

        features = self.movies.drop(columns=['MovieID', 'Title'])
        features['combined_text'] = features.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(features['combined_text'])
        similarity_matrix = cosine_similarity(tfidf_matrix)

        self.similarity_matrix_df = pd.DataFrame(similarity_matrix, index=self.movies['MovieID'], columns=self.movies['MovieID'])

        self.ratings = train_ratings.get_rating_matrix()
        self.ratings_means = self.ratings.mean()
        self.centralized_ratings = self.ratings.sub(self.ratings_means, axis=1)

        self.centralized_ratings.fillna(0, inplace=True)


    def predict(self, X_test):
        """
        Provide a Content-based Filtering Recommendations
        """
        temp_users = [user for user in X_test.get_users() if user in self.ratings.columns]
        predictions = pd.DataFrame(np.zeros((len(self.movies), len(temp_users))), index=self.movies['MovieID'], columns=temp_users)
        predictions.index.names = ['MovieID']
        predictions.columns.names = ['UserID']
        predictions.update(self.centralized_ratings)

        predictions = predictions.T

        similarity_matrix = self.similarity_matrix_df
        predicted_ratings = np.dot(predictions, similarity_matrix) / np.abs(similarity_matrix).sum(axis=1).values

        predicted_ratings_df = pd.DataFrame(predicted_ratings, index=predictions.index, columns=predictions.columns)

        predicted_ratings_df = predicted_ratings_df.T
        user_means = self.ratings_means.reindex(predicted_ratings_df.columns).values
        predicted_ratings_df = predicted_ratings_df.add(user_means, axis='columns')
        predicted_ratings_df = predicted_ratings_df.T

        predictions_df = predicted_ratings_df.reset_index().melt(id_vars='UserID', value_name='Rating')
        predictions_df.reset_index(drop=True, inplace=True)

        predictions_df = predictions_df.pivot(index='MovieID', columns='UserID', values='Rating')
        predictions_df = predictions_df.loc[X_test.get_movies()][temp_users]
        predictions_df = predictions_df.round() * X_test.get_rating_matrix()[temp_users]

        return RatingMatrix(predictions_df)
