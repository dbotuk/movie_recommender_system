import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack




def create_transition_matrix(matrix):
    """
    computes transition matrix needed for Page Rank 

    Inputs:
    - matrix: train matrix with users as columns, movies as rows and ratings as values

    Return:
    - transition_matrix: needed for Page Rank calculation
    """

    user_movie_matrix = csr_matrix(matrix)
    movie_user_matrix = user_movie_matrix.T
    user_identity_matrix = csr_matrix((user_movie_matrix.shape[1], user_movie_matrix.shape[1]))
    movie_identity_matrix = csr_matrix((user_movie_matrix.shape[0], user_movie_matrix.shape[0]))

    #stack the matrices to form a full adjacency matrix
    #top half will look like this:      user-identity | user-movie
    #bottom half will look like this:   movie-user    | movie-identity
    top = hstack([user_identity_matrix, movie_user_matrix])
    bottom = hstack([user_movie_matrix, movie_identity_matrix])
    adjacency_matrix = vstack([top, bottom])

    row_sums = np.array(adjacency_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 0.01
    D_inv = csr_matrix(np.diag(1.0 / row_sums))
    transition_matrix = D_inv @ adjacency_matrix

    return transition_matrix 



def personalized_page_rank(transition_matrix, personalization_vector, alpha=0.85, max_iter=300, tol=1e-6):
    """
    computes the Personalized Page Rank scores.

    Inputs:
    - transition_matrix: transition probability matrix.
    - personalization_vector: initial personalized vector (vector with an identifier of a user).
    - alpha: damping factor 
    - max_iter: maximum number of iterations to prevent infinity loop.
    - tol: threshold to check convergence.

    Return:
    - rank_vector: vector with a Page Rabk scores
    """
    
    n = transition_matrix.shape[0]
    rank_vector = np.array(personalization_vector, dtype=np.float64)
    rank_vector /= rank_vector.sum()
    
    for _ in range(max_iter):
        new_rank_vector = (alpha * transition_matrix.T @ rank_vector) + ((1 - alpha) * personalization_vector)
        if np.linalg.norm(new_rank_vector - rank_vector, 1) < tol:
            break
        rank_vector = new_rank_vector
    
    return rank_vector