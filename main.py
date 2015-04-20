#Visualization-of-Latent-Factors-from-Movies
import numpy as np
from scipy.sparse import coo_matrix
import random
import copy
from numpy.linalg import norm
from numpy.linalg import inv
import pickle as pk
from sklearn.decomposition import TruncatedSVD

def readMovie():
    ''' Read movie data-set from file.
    
    Returns:
        movie matrix: each column is the feature vector for a movie
        id_2_name: a dict of {movie_id: movie name}
        id_2_row: a dict of {movie_id: row number in the movie matrix}
                  Note we keep this dict mostly because Python starts index
                  from 0 while the data-set starts from id=0.
    '''

    # Initialize all dicts
    id_2_name = {}
    id_2_row = {}
    
    # {movie_id: feature vector}, to help construct movie matrix
    features = {}
    
    # Parsing from file, also construct id_2_name along the way
    with open('movies.txt', 'r') as fin:
        data = fin.readline().strip().split('\r')
        for line in data:
            line = line.split('\t')
            id = int(line[0])
            id_2_name[id] = line[1]
            features[id] = [float(d) for d in line[2:]]

    # Initialize movie matrix
    f_mat = []

    # Construct id_2_row and movie matrix
    for row, id in enumerate(sorted(features)):
        id_2_row[id] = row
        f_mat.append(features[id])

    return np.array(f_mat), id_2_name, id_2_row


def readUser(mid_2_col):
    ''' Read user data-set from file.
    Args:
        mid_2_col: a dict of {movie_id: column number in the rating matrix}
                   Note we keep this because Python starts index from 0 wihle
                   the data-set starts from 1.
    
    Returns:
        rating matrix: entry(i, j) means the rating of user i to movie j
        uid_2_row: a dict of {user_id: row number in the rating matrix}
                   Note we keep this because Python starts index from 0 wihle
                   the data-set starts from 1.
        obs: a list of (i, j), where entry (i, j) of the rating matrix is
             an observed rating instead of a zero entry.
    '''
    
    # Parse rating from data
    with open('data.txt', 'r') as fin:
        data = fin.readline().strip().split('\r')
        ratings = [[int(d) for d in line.split('\t')] for line in data]
    
    # Construct uid_2_row
    uid_2_row = {}
    for row, uid in enumerate(sorted(set([r[0] for r in ratings]))):
        uid_2_row[uid] = row

    # Construct rating matrix from sparse matrix
    # Also construct the list of observed entry coordinates (obs)
    obs = []
    rows = []
    cols = []
    data = []
    for r in ratings:
        uid, mid, rating = r
        rows.append(uid_2_row[uid])
        cols.append(mid_2_col[mid])
        data.append(float(rating))
        obs.append((uid_2_row[uid], mid_2_col[mid]))

    return coo_matrix((data, (rows, cols))).toarray(), uid_2_row, obs


def readUVab(fname):
    '''Reads computed U, V, a, b from file.
    
    Args:
        fname: a string, a pickle file name.
    
    Returns:
        U: latent factor matrix for users
        V: latent factor matrix for movies
        a: offset vector for users
        b: offset vector for movies
    '''
    
    with open(fname, 'r') as fin:
        data = pk.load(fin)
    return data['U'], data['V'], data['a'], data['b']


def saveUVab(fname, U, V, a, b):
    '''Saves computed U, V, a, b to file.
    
    Args:
        fname: a string, a file name to save to
        U: latent factor matrix for users
        V: latent factor matrix for movies
        a: offset vector for users
        b: offset vecto for movies
    '''

    with open(fname, 'w') as fout:
        pk.dump({'U': U, 'V': V, 'a': a, 'b': b}, fout)


def svd2d(V, U):
    '''Performs singular value decomposition on V, and apply 2D projection
    on V and U. We basically uses sklearn's API.
    
    Args:
        V: the latent matrix for movies
        U: the latent matrix for users
    
    Returns:
        tsvd: sklearn's svd object that has already trained on V
        V2: projected V in 2D
        U2: projected U in 2D
    '''
    tsvd = TruncatedSVD()
    tsvd.fit(V)
    return tsvd, tsvd.transform(V), tsvd.transform(U)


def sgd(Y, M, N, K, obs, lmb=.001, rate=.001, decay=.8):
    '''Performs stochastic gradient descent for matrix factorization.
    
    Args:
        Y: the M by N ratings matrix
        M: number of users
        N: number of movies
        K: number of latent factors
        obs: a list of observed rating coordinate, i.e. if (i, j) in obs,
             then Y[i, j] is a non-zero entry / valid rating.
        lmb: regularization coefficient
        rate: initial learning rate
        decay: the change in learning rate after each iteration
    
    Returns:
        U: latent factor matrix for users (K by N)
        V: latent factor matrix for movies (K by M)
        a: offset vector for users (dim M)
        b: offset vector for movies (dim N)
    '''
    # Threshold for convergence
    eps = .001
    # Max number of iterations
    max_iter = 1000
    # Current iteration index
    n = 0

    # Initilize each U, V, a, b
    new_U = np.random.rand(M, K)
    new_V = np.random.rand(N, K)
    new_a = np.random.rand(M)
    new_b = np.random.rand(N)
    
    U = np.zeros((M, K))
    V = np.zeros((N, K))
    a = np.zeros(M)
    b = np.zeros(N)
    
    while True:
        # Check whether we have converged
        if norm(U - new_U)/rate < eps and \
            norm(V - new_V)/rate < eps and \
            norm(a - new_a)/rate < eps and \
            norm(b - new_b)/rate < eps:
            return U, V, a, b
        
        # Check whether we have reach max number of iterations
        n += 1
        if n > max_iter:
            return U, V, a, b

        # Update U, V, a, b
        U = new_U.copy()
        V = new_V.copy()
        a = new_a.copy()
        b = new_b.copy()
        
        # Shuffle the coordinates of observed ratings
        # (for stochastic gradient descent)
        list_ij = copy.deepcopy(obs)
        random.shuffle(list_ij)

        # For each coordinate in the shuffled list
        for i,j in list_ij:
            # Calculate the partial gradient
            du_i = lmb * new_U[i, :] - \
                (Y[i, j] - np.dot(new_U[i, :].T, new_V[j, :]) - a[i] - b[j]) \
                * new_V[j, :] * 2
            dv_j = lmb * new_V[j, :] - \
                (Y[i, j] - np.dot(new_U[i, :].T, new_V[j, :]) - a[i] - b[j]) \
                * new_U[i, :] * 2
            da_i = 2 * Y[i, j] - 2 * np.dot(U[i, :].T, new_V[j, :]) \
                - 2 * a[i] - 2 * b[j]
            db_j = 2 * Y[i, j] - 2 * np.dot(U[i, :].T, new_V[j, :]) \
                - 2 * a[i] - 2 * b[j]

            # Update new_U, new_V, new_a, new_b
            new_U[i, :] -= rate * du_i
            new_V[j, :] -= rate * dv_j
            new_a[i] -= rate * da_i
            new_b[j] -= rate * db_j

        # Update learning rate
        rate *= decay

def err(Y, Ypred, obs):
    '''Calculates the absolute error rate of the prediction matrix.
    
    Args:
        Y: the original rating matrix
        Ypred: the predicted rating matrix
        obs: the list of observed rating coordinates
    
    Returns:
        the average absolute error (abs(value - pred)) for each observed entry.
    '''
    
    sum = 0.0
    for i, j in obs:
        sum += np.abs(Y[i, j] - Ypred[i, j])
    return sum / len(obs)

def estimate(U, V, a, b, M, N):
    '''Calcluates the predicted ratings matrix
    
    Args:
        U: latent factor matrix for users
        V: latent factor matrix for movies
        a: offset vector for users
        b: offset vector for movies
        M: number of users
        N: number of movies
    
    Returns:
        Predicted rating matrix
    '''
    
    return np.dot(U, V.T) + np.array([a] * N).T + np.array([b] * M)


if __name__ == '__main__':
    # First read from file
    movies, mid_2_name, mid_2_col = readMovie()
    Y, uid_2_row, obs = readUser(mid_2_col)
    
    # Number of movies
    N = len(mid_2_col)
    # Number of users
    M = len(uid_2_row)
    # Number of latent factors
    K = 20
    
    # Runs matrix factoriztion
    U, V, a, b = sgd(Y, M, N, K, obs)
    # Saves the matrices to file
    save('UV.pk', U, V, a, b)
    
    # Predicts the rating matrix, and prints out the error rate
    YY = estimate(U, V, a, b, M, N)
    print err(Y, YY, obs)

    # Performs SVD on V and U
    tsvd, VP, UP = svd2d(V, U)

    # Do some data customized data analysis and visualization