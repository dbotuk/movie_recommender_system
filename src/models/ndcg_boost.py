import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import ndcg_score
from sklearn.base import BaseEstimator, RegressorMixin


class NDCGLossGradientBoostingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = GradientBoostingRegressor(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def ndcg_loss(self, y_true, y_pred):
        y_true = np.asarray(y_true).reshape(1, -1)
        y_pred = np.asarray(y_pred).reshape(1, -1)
        return 1 - ndcg_score(y_true, y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return -self.ndcg_loss(y, y_pred)