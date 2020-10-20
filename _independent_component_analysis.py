from _util import whitening, time_lagged_autocov, joint_diagonalization
import numpy as np
import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader

class SOBI(TransformerMixin, BaseEstimator):
    
    """
    
    Linear ICA for time series data using joint diagonalization of the lagged-autocovariance matrices
    
    """
    
    def __init__(self, lags=1, eps=1e-3, max_iter=1000):
        
        """
        
        Attributes:
            * lags: number of lags to consider (default=1)
            * eps: tolerance for stopping criteria (default=1e-3)
            * max_iter: maximum number of iterations taken for the solvers to converge (default=1000)

        """
        
        self.lags = lags
        self.eps = eps
        self.max_iter = max_iter
        self.is_fitted_ = False
    
    def fit(self, X):
        
        """
        
        Attributes:
            * X: time series data (dimension: time x variables)
            * lags: number of lags to consider (default=1)
            * eps: tolerance for stopping criteria (default=1e-3)
            * max_iter: maximum number of iterations taken for the solvers to converge (default=1000)

        """

        X_white, U, d = whitening(X.T)
        C = time_lagged_autocov(X_white, self.lags)
        C = C + 1J*np.zeros_like(C)
        V, C = joint_diagonalization(C, eps=self.eps, max_iter=self.max_iter)
        self.W = (V.T).dot((U / d).T)
        
        self.is_fitted_ = True

    def transform(self, X):
        
        """
        
        Attributes:
            * X: time series data (dimension: time x variables)

        Returns:
            * Estimated sources

        """
        
        check_is_fitted(self, 'is_fitted_')
        return np.real(X.dot(self.W.T))
    
    def fit_transform(self, X):
        
        """
        
        Attributes:
            * X: time series data (dimension: time x variables)
            * lags: number of lags to consider (default=1)
            * eps: tolerance for stopping criteria (default=1e-3)
            * max_iter: maximum number of iterations taken for the solvers to converge (default=1000)

        Returns:
            * Estimated sources

        """
        
        self.fit(X)
        return self.transform(X)
    

class SlowFlows:
    
    """
    
    Non-linear ICA for time series data using slow-flows
    Source: https://arxiv.org/pdf/2007.10182.pdf
    
    """
    
    def __init__(self, D, flowModel=None, final_ica=None, device=torch.device('cpu')):
        
        """
        
        Attributes:
            * D: dimension of the input (number of variables)
            * flowModel: embedding flow-based model. Can be defined outside the SlowFlows class.
                         If None, defined inside.
            * final_ica: ica applied on the output of the flowModel. Can be any ICA model with 
                         **fit** and **transform** methods. If None, uses SOBI ICA defined in
                         SOBI.py module.
            * device: device on which computation is done. Compatible with Nvidia GPUs. 
        
        """
        
        self.D = D
        
        if flowModel is None:
            NN = lambda: nn.Sequential(nn.Linear(D//2, 16), nn.LeakyReLU(), nn.Linear(16, D))
            self.flowModel = FlowModel(D, NN, 12).to(device)
        else:
            self.flowModel = flowModel.to(device)
        
        self.final_ica = final_ica
        if self.final_ica is None:
            self.final_ica = SOBI(lags=100, eps=1e-3, max_iter=1000)
            
        self.device = device
        self.sf_is_fitted_ = False
        self.ica_is_fitted_ = False
        
    def fit(self, X, batch_size=1, lr=1e-3, num_epochs=1500, verbose=True):
        
        """
        
        Attributes:
            * X: whole time series data, with shape (samples x time x variables)
            * batch_size: number of samples per batch
            * lr: learning rate
            * num_epochs: number of training epochs
            * verbose: set True for training verbosity
        
        """
        
        optimizer = torch.optim.Adam(self.flowModel.parameters(), lr=lr)
        
        if len(X.size()) == 3:
            batch_size = batch_size
            shuffle = True
        else:
            raise print('Wrong data shape, expected (batch x time x variables), got {}'.format(X.size()))
        
        index = list(range(batch_size))    
        dataloader = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=shuffle)
        
        for epoch in range(num_epochs):
        
            loss_epoch = []
            loss = None
            
            for x_batch in dataloader:
                optimizer.zero_grad()

                x = x_batch.reshape(-1, x_batch.size(-1)).to(self.device)
                
                # Since the model is symmetric (Gaussian variables and increments) we can 
                # augment the data by inversing the series.
                
                index = np.arange(x.size(0))
                if epoch%2==0:
                    index = index[::-1]

                x = x[list(index)]

                previous_loss = torch.zeros(x.size(0)).to(self.device)
                z, logprob = self.flowModel.to_embedding(x, previous_loss)
                
                if loss is None:
                    loss = -logprob.mean()
                else:
                    loss += -logprob.mean()

                z = z.reshape(x_batch.size())
                
                for i, s in enumerate(self.flowModel.scales):
                    s = s**2
                    m = MultivariateNormal(torch.zeros(1, self.D).to(self.device), torch.diag(s))
                    if i == 0:
                        z_lag = z
                    else:
                        z_lag = z[:, i:] - z[:, :-i]
                    loss -= m.log_prob(z_lag).mean()
            
            loss /= len(dataloader)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                self.flowModel.layers[-1].W.div_(torch.norm(self.flowModel.layers[-1].W))
            
            loss_epoch.append(loss.item())
            
            if verbose and epoch % 10 == 0:
                print('iter %s:' % epoch, 'loss = %.3f' % np.mean(loss_epoch))
                
        self.sf_is_fitted_ = True
        
        x = X.reshape(-1, X.size(-1)).to(self.device)
        previous_loss = torch.zeros(x.size(0)).to(self.device)
        z, logprob = self.flowModel.to_embedding(x, previous_loss)
        self.final_ica.fit(z.cpu().detach().numpy())
        
        self.ica_is_fitted_ = True
        
    def transform(self, X):
        
        """
        
        Attributes:
            * X: time series data (dimension: time x variables)

        Returns:
            * Estimated sources

        """
        
        X = X.to(self.device)
        
        if self.sf_is_fitted_:
            x = X.reshape(-1, X.size(-1))
            previous_loss = torch.zeros(x.size(0)).to(self.device)
            z, logprob = self.flowModel.to_embedding(x, previous_loss)
        else:
            raise print('Fit before transforming.')
        
        if self.ica_is_fitted_:
            return self.final_ica.transform(z.cpu().detach().numpy())
        else:
            self.ica_is_fitted_ = True
            return self.final_ica.fit_transform(z.cpu().detach().numpy())
    
    def fit_transform(self, X, batch_size=None, lr=1e-3, num_epochs=1500, verbose=True):
        
        """
        
        Attributes:
            * X: time series data (dimension: time x variables)
            * lags: number of lags to consider (default=1)
            * eps: tolerance for stopping criteria (default=1e-3)
            * max_iter: maximum number of iterations taken for the solvers to converge (default=1000)

        Returns:
            * Estimated sources

        """
        
        X = X.to(self.device)
        
        self.fit(X, batch_size, lr, num_epochs, verbose)
        return self.transform(X)
    
