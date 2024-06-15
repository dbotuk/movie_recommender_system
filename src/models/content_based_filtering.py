import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import RatingMatrix


class ContentBasedFilteringRecommender:
    def __init__(self):
        self.ratings_means = None
        self.ratings = None
        self.similarity_matrix = None
        self.recommendations = pd.DataFrame()

    def train(self, train_ratings, movies):
        """
        Train a Content-based Filtering model
        """
        features = movies.drop(columns=['MovieID', 'Title'])
        features['combined_text'] = features.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(features['combined_text'])

        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        self.similarity_matrix = pd.DataFrame(self.similarity_matrix, index=movies['MovieID'], columns=movies['MovieID'])

        self.ratings = train_ratings.get_rating_matrix()
        self.ratings_means = self.ratings.mean()
        self.ratings = self.ratings.sub(self.ratings_means, axis=1)

        self.ratings.fillna(0, inplace=True)

    def predict(self, X_test, users, movies):
        """
        Provide a Content-based Filtering Recommendations
        """
        temp_users = [user for user in users if user in self.ratings.columns]
        temp_movies = [movie for movie in movies if movie in self.ratings.index]
        temp_rec = self.ratings.loc[temp_movies][temp_users].T
        temp_means = self.ratings_means[temp_users]

        temp_similarity = self.similarity_matrix.loc[temp_movies][temp_movies]

        predicted_ratings = np.dot(temp_rec, temp_similarity) / np.abs(temp_similarity).sum(axis=1).values

        predicted_ratings += np.tile(temp_means.values, (predicted_ratings.shape[1], 1)).astype('float64').T
        predicted_ratings = np.clip(predicted_ratings, 1, 5)

        predicted_ratings_df = pd.DataFrame(predicted_ratings, index=temp_rec.index, columns=temp_rec.columns)
        predictions_df = predicted_ratings_df.reset_index().melt(id_vars='UserID', value_name='Rating')
        predictions_df.reset_index(drop=True, inplace=True)

        predictions_df = predictions_df.pivot(index='MovieID', columns='UserID', values='Rating')
        predictions_df = predictions_df * X_test.get_rating_matrix()

        return RatingMatrix(predictions_df)
