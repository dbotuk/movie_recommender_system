import numpy as np


def map_score(y_true, y_pred, top=10):
    """
    Calculate Mean Average Precision
    """
    result = 0
    for user in y_pred.get_rating_matrix().columns:
        top_pred = y_pred.get_user_ratings(user).sort_values(ascending=False).head(top).index
        top_true = y_true.get_user_ratings(user).sort_values(ascending=False).head(top).index
        result += average_precision(top_true, top_pred)
    return result / len(y_pred.get_rating_matrix().columns)


def average_precision(true_rec, pred_rec):
    """
    Calculate Average Precision for User Recommendation
    """
    return sum([precision_at_k(true_rec, pred_rec, top) for top in range(len(true_rec))]) / len(true_rec)


def precision_at_k(true_rec, pred_rec, top):
    """
    Calculate Precision at k Position
    """
    return len(set(true_rec[:top+1]) & set(pred_rec[:top+1])) / (top+1)


def mrr_score(y_true, y_pred, top=10):
    """
    Calculate Mean Reciprocal Rank
    """
    scores = []
    for user in y_pred.get_rating_matrix().columns:
        sorted_pred = y_pred.get_user_ratings(user).sort_values(ascending=False).head(top).index
        sorted_true = y_true.get_user_ratings(user).sort_values(ascending=False).head(top).index
        scores.append(reciprocal_rank(sorted_pred, sorted_true))
    return np.mean(scores)


def reciprocal_rank(true_rec, pred_rec):
    """
    Calculate Reciprocal Rank
    """
    for i, item in enumerate(pred_rec):
        if item in true_rec:
            return 1 / (i + 1)
    return 0


def ndcg_score(y_true, y_pred, top=10):
    """
    Calculate the Normalized Discounted Cumulative Gain (NDCG)
    """
    ndcg_values = []
    for user in y_pred.get_rating_matrix().columns:
        top_pred = y_pred.get_user_ratings(user).sort_values(ascending=False).head(top).index
        top_true = y_true.get_user_ratings(user).sort_values(ascending=False).head(top).index

        dcg_value = discounted_cumulative_gain(top_true, top_pred)
        idcg_value = ideal_discounted_cumulative_gain(top_true)

        ndcg_value = dcg_value / idcg_value if idcg_value > 0 else 0

        ndcg_values.append(ndcg_value)

    return np.mean(ndcg_values)


def discounted_cumulative_gain(true_rec, pred_rec):
    """
    Calculate the Discounted Cumulative Gain (DCG)
    """
    dcg_value = 0.0
    for i, movie in enumerate(pred_rec):
        if movie in true_rec:
            dcg_value += 1 / np.log2(i + 2)
    return dcg_value


def ideal_discounted_cumulative_gain(true_rec):
    """
    Calculate the Ideal Discounted Cumulative Gain (IDCG)
    """
    return sum([1 / np.log2(i + 2) for i in range(len(true_rec))])


def rmse_score(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted ratings
    """
    y_pred = y_pred.get_rating_matrix()
    y_true = y_true.get_rating_matrix()
    return np.sqrt(np.nansum((y_pred[y_true.notna()] - y_true[y_true.notna()])**2))