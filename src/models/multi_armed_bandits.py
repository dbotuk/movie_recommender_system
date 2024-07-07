import numpy as np



class EpsilonGreedyBandit:
    def __init__(self, rating_matrix, epsilon=0.1):
        self.rating_matrix = rating_matrix
        self.epsilon = epsilon
        self.n_arms = len(rating_matrix.get_movies())
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.movie_indices = {movie_id: idx for idx, movie_id in enumerate(rating_matrix.get_movies())}
        self.inverse_movie_indices = {idx: movie_id for movie_id, idx in self.movie_indices.items()}

    def select_arm(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

    def recommend(self, user_id):
        chosen_arm = self.select_arm()
        movie_id = self.inverse_movie_indices[chosen_arm]
        return movie_id

    def fit(self, train_data):
        for user_id in train_data.get_users():
            user_ratings = train_data.get_user_ratings(user_id).dropna()
            for movie_id, rating in user_ratings.items():
                if movie_id in self.movie_indices:
                    chosen_arm = self.movie_indices[movie_id]
                    self.update(chosen_arm, rating)

    def evaluate(self, test_data):
        total_reward = 0
        n_recommendations = 0
        for user_id in test_data.get_users():
            user_ratings = test_data.get_user_ratings(user_id).dropna()
            for movie_id, actual_rating in user_ratings.items():
                if movie_id in self.movie_indices:
                    recommended_movie = self.recommend(user_id)
                    if recommended_movie == movie_id:
                        total_reward += actual_rating
                    n_recommendations += 1
        return total_reward / n_recommendations if n_recommendations > 0 else 0



class UCBBandit:
    def __init__(self, rating_matrix, alpha=1):
        self.rating_matrix = rating_matrix
        self.alpha = alpha
        self.n_arms = len(rating_matrix.get_movies())
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.movie_indices = {movie_id: idx for idx, movie_id in enumerate(rating_matrix.get_movies())}
        self.inverse_movie_indices = {idx: movie_id for movie_id, idx in self.movie_indices.items()}

    def select_arm(self):
        total_counts = np.sum(self.counts)
        bonus = np.sqrt((2 * np.log(total_counts + 1)) / (self.counts + 1e-5))
        ucb_values = self.values + self.alpha * bonus
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[chosen_arm] = new_value

    def recommend(self, user_id):
        chosen_arm = self.select_arm()
        movie_id = self.inverse_movie_indices[chosen_arm]
        return movie_id

    def fit(self, train_data):
        for user_id in train_data.get_users():
            user_ratings = train_data.get_user_ratings(user_id).dropna()
            for movie_id, rating in user_ratings.items():
                if movie_id in self.movie_indices:
                    chosen_arm = self.movie_indices[movie_id]
                    self.update(chosen_arm, rating)

    def evaluate(self, test_data):
        total_reward = 0
        n_recommendations = 0
        for user_id in test_data.get_users():
            user_ratings = test_data.get_user_ratings(user_id).dropna()
            for movie_id, actual_rating in user_ratings.items():
                if movie_id in self.movie_indices:
                    recommended_movie = self.recommend(user_id)
                    if recommended_movie == movie_id:
                        total_reward += actual_rating
                    n_recommendations += 1
        return total_reward / n_recommendations if n_recommendations > 0 else 0
