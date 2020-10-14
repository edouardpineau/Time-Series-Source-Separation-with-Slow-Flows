import numpy as np
import itertools

def center(X, mean=None):
    
    if mean is None:
        return X - X.mean(axis=1, keepdims=True)
    else:
        return X - mean
    
def time_lagged_autocov(X, lags):
    
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

def whitening(X, C=None):
    
    if C is None:
        C = X.shape[0]
        
    X = center(X)
    U, d, _ = np.linalg.svd(X, full_matrices=False)
    U_d = (U / d).T[:C]
    X_whiten = np.dot(U_d, X) * np.sqrt(X.shape[1]) # np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    
    return X_whiten, U, d
    
def off_frobenius(M):
    
    return (np.linalg.norm(np.tril(M, k=-1), ord='fro')**2 + np.linalg.norm(np.triu(M, k=1), ord='fro')**2)

def rotation(M):
    
    h = np.array([M[:, 0, 0] - M[:, 1, 1], 
                  M[:, 1, 0] + M[:, 0, 1], 
                  1j*(M[:, 1, 0] - M[:, 0, 1])]).T
    # G = np.stack([np.expand_dims(h_, axis=1).dot(np.expand_dims(h_, axis=0)) for h_ in h]).sum(0)
    G = np.real(h.T.dot(h))
    [eigvals,v] = np.linalg.eigh(G)
    [x, y, z] = np.sign(v[0, -1])*v[:,-1]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    c = np.sqrt((x + r) / (2*r))
    s = (y - 1j*z) / np.sqrt(2*r*(x + r))
    
    R = np.array([[c, np.conjugate(s)], [-s, np.conjugate(c)]])
    
    return R

def joint_diagonalization(C, V=None, eps=1e-3, max_iter=1000, verbose=-1):
    
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

def ICA(X, lag=1, eps=1e-3, max_iter=1000):
    
    X_white, U, d = whitening(X)
    C = time_lagged_autocov(X_white, lag)
    C = C + 1J*np.zeros_like(C)
    V, C = joint_diagonalization(C, eps=eps, max_iter=max_iter)
    W = (V.T).dot((U / d).T)
    S = np.real(W.dot(X))
    
    return S