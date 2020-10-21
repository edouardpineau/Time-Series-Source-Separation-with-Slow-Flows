import numpy as np
import pandas as pd
import itertools


def to_dataframe(array):
    """ Transform array to pandas.DataFrame()

    Parameters
    ----------
    array : ndarray, Iterable, dict
        array to transform to DataFrame

    Returns
    -------
    pandas.DataFrame
        Transformed DataFrame
    """
    if not isinstance(array, pd.DataFrame):
        return pd.DataFrame(array)
    else:
        return array


def to_series(array):
    """ Transform array to pandas.Series()

    Parameters
    ----------
    array : array-like, Iterable, dict, or scalar value
        array to transform to Series

    Returns
    -------
    pandas.Series
        Transformed Series
    """
    if not isinstance(array, pd.Series):
        return pd.Series(array)
    else:
        return array


def get_triu(df, return_1d=True):
    """ Upper triangle of an array excluding diagonal values.

    Examples
    --------
    >>> df = pd.DataFrame([[1, 2, 3,], [4, 5, 6], [7, 8, 9]])
    >>> get_triu(df)

    |        |   0 |
    |:-------|----:|
    | (0, 1) |   2 |
    | (0, 2) |   3 |
    | (1, 2) |   6 |

    >>> get_triu(df, return_1d=False)

    |    |   0 |   1 |   2 |
    |---:|----:|----:|----:|
    |  0 | nan |   2 |   3 |
    |  1 | nan | nan |   6 |
    |  2 | nan | nan | nan |

    Parameters
    ----------
    df : pandas.DataFrame()
        Target 2d data
    return_1d : bool, optional
        Whether to squeeze to 1d, by default True

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Upper triangle values excluding diagonal
    """
    mask = np.tril(np.ones(df.shape))
    mask = mask.astype(bool)
    df = df.mask(mask)

    if return_1d:
        return df.stack()
    else:
        return df


def center(X, mean=None):
    
    """
    Function to center the data using empirical mean to have each variable 
    with zero mean.
    
    Attributes:
        * X: data to center
        * mean: if the true mean is known, can be used (default=None)
    
    Returns: 
        * Centered data
    """
    
    if mean is None:
        return X - X.mean(axis=1, keepdims=True)
    else:
        return X - mean
        
        
def whitening(X):
    
    """
    Function that withens the data.
    
    Attributes:
        * X: time series data
        
    Returns:
        * Whitened data
        * Singular vectors
        * singular values
    
    """
    
    X = center(X)
    U, d, _ = np.linalg.svd(X, full_matrices=False)
    U_d = (U / d).T
    X_whiten = np.dot(U_d, X) * np.sqrt(X.shape[1])
    
    return X_whiten, U, d
    
    
def time_lagged_autocov(X, lags):
    
    """
    Computes the auto-covariance tensor, containing all lagged-autocovariance 
    with lag from 0 (covariance) to lags
    
    Attributes:
        * X: time series data (dimension: variables x time)
        * lags: number of lags to consider
        
    Returns:
        * Autocovariance tensor
        
    """
    
    lags = lags + 1
    n, l = X.shape
    L = l - lags
    R = np.empty([lags, n, n])
    
    X0 = center(X[:, :L])
    
    for k in range(lags):
        Xk = center(X[:, k:k+L])
        R[k] = (1.0/L)*(X0.dot(Xk.T))
        R[k] = 0.5*(R[k] + R[k].T)
    
    return R
    
    
def off_frobenius(M):
    
    """
    Computes the square Frobenius norm of the matrix M-diag(M)
    
    Attributes:
        * M: square matrix
        
    Returns:
        * Off-diagonal Frobenius norm
    
    
    """
    
    return (np.linalg.norm(np.tril(M, k=-1), ord='fro')**2 + np.linalg.norm(np.triu(M, k=1), ord='fro')**2)

def rotation(M):
    
    """
    This function infers Jacobi rotation matrix R used in the joint 
    diagonalization of a set of matrices
    
    See: https://en.wikipedia.org/wiki/Jacobi_rotation
    
    Attributes:
        * M: matrix to be rotated
        
    Returns:
        * Rotation matrix
    
    
    """
    
    h = np.array([M[:, 0, 0] - M[:, 1, 1], 
                  M[:, 1, 0] + M[:, 0, 1], 
                  1j*(M[:, 1, 0] - M[:, 0, 1])]).T
    G = np.real(h.T.dot(h))
    [eigvals,v] = np.linalg.eigh(G)
    [x, y, z] = np.sign(v[0, -1])*v[:,-1]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    c = np.sqrt((x + r) / (2*r))
    s = (y - 1j*z) / np.sqrt(2*r*(x + r))
    
    R = np.array([[c, np.conjugate(s)], [-s, np.conjugate(c)]])
    
    return R

def joint_diagonalization(C, V=None, eps=1e-3, max_iter=1000, verbose=-1):
    
    """
    Joint diagonalization of a set of matrices C
    
    Attributes:
        * C: set of symmetric matrices
        * V:  a priori eigan-vectors of the matrices C (default=None)
        * eps: tolerance for stopping criteria (default=1e-3)
        * max_iter: maximum number of iterations taken for the solvers to 
          converge (default=1000)
    
    Returns:
        * V: a posteriori eigen-vectors of the matrices C
        * C: diagonalized matrices
        
    """
    
    d = C.shape[1]
    list_pairs = list(itertools.combinations(range(d), 2))
    
    if V is None:
        V = np.eye(d) + 1j*np.zeros((d, d))

    O_cs = np.sum([off_frobenius(c) for c in C])
    counter = 0
    
    if verbose > 0:
        print('Iter: {:.0f}, Diagonalization: {:.2f}'.format(counter, O_cs))
        
    diff = np.inf
    
    while ((diff > eps) and (counter < max_iter)):
        counter += 1
        for (i,j) in list_pairs:
            V_ = np.eye(d) + 1j*np.zeros((d, d))
            idx = (slice(None), ) + np.ix_([i,j],[i,j])
            R = rotation(C[idx])
            V_[np.ix_([i,j],[i,j])] = V_[np.ix_([i,j],[i,j])].dot(R)
            V = V.dot(V_.T)
            C = np.matmul(np.matmul(V_, C), V_.T)

        O_cs_new = np.sum([off_frobenius(c) for c in C])
        diff = np.abs(O_cs - O_cs_new)
    
        if verbose > 0:
            print('Iter: {:.0f}, Diagonalization: {:.2f}'.format(counter, O_cs))
        O_cs = O_cs_new
        
    return V, C