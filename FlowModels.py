import torch
import torch.nn as nn

import math
import numpy as np


class Layer(nn.Module):
    
    """ **Layer** abstract class. We remind that an abstract class is used as a blueprint for other classes. It allows you to create a set of methods that must be created within any child classes built from the abstract class.
    We note that the standard way to used abstract classes in Python is to used ABC module. Not here.
    
        Methods
        -------
            * to_embedding: from data to latent representation
            * from_embedding: from latent representation to data
            
    We note that in flow-based models, the **from_embedding** is the exact inverse of the **to_embedding**.
    """
    
    def __init__(self):
        super(Layer, self).__init__()
        
    def to_embedding(self, z, previous_loss):
        raise NotImplementedError
    
    def from_embedding(self, x, previous_loss):
        raise NotImplementedError


class LayerList(Layer, nn.Module):
    
    """ **LayerList** is a list of layers from class **Layer**. We remark that the methods have same input and output format shape: we can chain layers. 
        
        Parameters
        ----------
            * list_of_layers: list of layers
            
        Methods
        -------
            * to_embedding: from data to latent representation through chain of layers
            * from_embedding: from latent representation to data though chain of inverse layers
    """
    
    def __init__(self, list_of_layers=None):
        super(LayerList, self).__init__()
        self.layers = nn.ModuleList(list_of_layers)

    def __getitem__(self, i):
        return self.layers[i]

    def to_embedding(self, x, previous_loss):
        for layer in self.layers:
            x, previous_loss = layer.to_embedding(x, previous_loss)
        return x, previous_loss

    def from_embedding(self, x, previous_loss):
        for layer in reversed(self.layers): 
            x, previous_loss = layer.from_embedding(x, previous_loss)
        return x, previous_loss


class ActivationNormalization(Layer):
    
    """ **ActivationNormalization** implements Activation Normalization (ActNorm), a type of normalization used for flow-based generative models. It was introduced in the GLOW architecture. An ActNorm layer performs an affine transformation of the activations using a scale and bias parameter per channel, similar to batch normalization.
    
        Parameters
        ----------
            * D: dimension of the input
            * rho: parameter of the moving average for the normalization parameters update
            
    """
    
    def __init__(self, D, rho=0.99):
        super(Layer, self).__init__()
        self.D = D
        
        self.logs_ = nn.Parameter(torch.zeros(1, D), requires_grad = True)
        self.b_ = nn.Parameter(torch.zeros(1, D), requires_grad = True)
        
        self.running_mean_ = nn.Parameter(torch.zeros(1, D), requires_grad = False)
        self.running_std_ = nn.Parameter(torch.ones(1, D), requires_grad = False)
        
        self.initialized = False
        self.rho = 0.99
        
    def to_embedding(self, x, previous_loss):         
        
        self.batch_mean_ = x.mean(0, keepdim=True)
        self.batch_std_ = x.std(0, keepdim=True)
        
        if not self.initialized:
            self.running_mean_.data.copy_(self.batch_mean_.data)
            self.running_std_.data.copy_(self.batch_std_.data)
            self.initialized = True

        if self.training:
            self.running_mean_.mul_(1 - self.rho)
            self.running_mean_.add_(self.rho * self.batch_mean_.data)

            self.running_std_.mul_(1 - self.rho)
            self.running_std_.data.add_(self.rho * self.batch_std_.data)

            mean_ = self.batch_mean_
            std_ = self.batch_std_

        else:
            mean_ = self.running_mean_
            std_ = self.running_std_
        
        output = (x - mean_) / (std_ + 1e-12)
        output = torch.exp(self.logs_) * output + self.b_
        previous_loss += torch.sum(self.logs_, dim=1)
        previous_loss -= torch.log(std_).sum(dim=-1)
        
        return output, previous_loss

    def from_embedding(self, x, previous_loss):
        assert self.initialized
        
        if self.training:
            mean_ = self.batch_mean_
            std_ = self.batch_std_
        else:
            mean_ = self.running_mean_
            std_ = self.running_std_
            
        output = (x - self.b_) / torch.exp(self.logs_)
        output = output * std_ + mean_
        previous_loss -= torch.sum(self.logs_, dim=1)
        previous_loss += torch.log(std_).sum(dim=-1)
        
        return output, previous_loss


class AffineCoupling(Layer):
    
    """ **AffineCoupling** implements affine coupling, a typical flow-based generative models layer. It was introduced in the RealNVP architecture. 
    
        Parameters
        ----------
            * D: dimension of the input
            * NN: neural networks used in the layer
            
    """
    
    def __init__(self, D, NN):
        super(AffineCoupling, self).__init__()
        self.NN = NN
        self.D = D
        
    def to_embedding(self, x, previous_loss):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        t = h[:, :self.D//2]
        s = torch.sigmoid(h[:, self.D//2:])
        z2_ = z2 * s + t
        previous_loss += torch.sum(torch.log(s), dim=1)
        return torch.cat([z1, z2_], dim=1), previous_loss
    
    def from_embedding(self, x, previous_loss):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        t = h[:, :self.D//2]
        s = torch.sigmoid(h[:, self.D//2:])
        z2_ = (z2 - t) / s
        previous_loss -= torch.sum(torch.log(s), dim=1)
        return torch.cat([z1, z2_], dim=1), previous_loss


class Linear(Layer):
    
    """ **Linear** implements a full-rank linear Layer. It was introduced in the GLOW architecture. 
    
        Parameters
        ----------
            * D: dimension of the input
    """
    
    def __init__(self, D):
        super(Linear, self).__init__()
        self.D = D
        W = np.random.randn(D, D)
        W = np.linalg.qr(W)[0].astype(np.float32)  # Orthogonal initialization
        self.W = nn.Parameter(torch.from_numpy(W))
        self.initialized = False
        
    def to_embedding(self, z, previous_loss):
        
        if not self.initialized:
            _, s, self.W.data = torch.svd(z)
            self.initialized = True
            
        previous_loss += torch.slogdet(self.W)[1] * self.D
        return torch.mm(z, self.W), previous_loss
    
    def from_embedding(self, z, previous_loss):
        W = torch.inverse(self.W.double()).float()
        previous_loss -= torch.slogdet(self.W)[1] * self.D
        return torch.mm(z, W), previous_loss


class Prior(Layer):
    
    """ **Prior** is the prior used for the latent space. Here, N(mean, var).  
    
        Parameters
        ----------
            * mean: mean of the latent distribution
            * logvar: log-variance of the latent distribution
    """
    
    def __init__(self, mean, logvar):
        super(Prior, self).__init__()
        self.mean = mean
        self.logvar = logvar

    def to_embedding(self, z, previous_loss):
        return z, -(z**2).sum(1)/2 - math.log((2*math.pi))*self.mean.size(-1)/2 + previous_loss
    
    def from_embedding(self, z, previous_loss):
        return z, previous_loss
    
    def sample_(self, n_sample):
        eps = torch.cat([torch.FloatTensor(self.prior.mean.size(-1)).normal_() for k in range(10)], dim=0)
        return self.mean + torch.exp(self.logvar/2) * eps


class Shuffle(Layer):
    
    """ **Shuffle** shuffles the dimensions on the layers output. 
    
        Parameters
        ----------
            * D: dimension of the input
    """
    
    
    def __init__(self, D):
        super(Shuffle, self).__init__()
        self.shuffled = False
        indices = np.arange(D)
        
        while (indices==np.arange(D)).astype(int).sum().astype(bool):
            np.random.shuffle(indices)
        
        rev_indices = np.zeros_like(indices)
        for i in range(D): 
            rev_indices[indices[i]] = i

        self.indices = torch.from_numpy(indices).long()
        self.rev_indices = torch.from_numpy(rev_indices).long()

    def to_embedding(self, x, previous_loss):
        if self.shuffled:
            self.shuffled = False
            return x[:, self.rev_indices], previous_loss
        else:
            self.shuffled = True
            return x[:, self.indices], previous_loss

    def from_embedding(self, x, previous_loss):
        if self.shuffled:
            self.shuffled = False
            return x[:, self.rev_indices], previous_loss
        else:
            self.shuffled = True
            return x[:, self.indices], previous_loss

        
class Lag(Layer):
    
    """ **Lag** implements a lag layer for the output of the slow-flow.
    
        Parameters
        ----------
            * D: dimension of the input
            * lag: size of the lag
    """
    
    def __init__(self, D, lag=1):
        super(Lag, self).__init__()
        self.lag = lag
        
    def to_embedding(self, z, previous_loss):
        return torch.cat([z[:self.lag], z[self.lag:] - z[:-self.lag]], dim=0), previous_loss
        
    def from_embedding(self, z, previous_loss):
        z_delagged = z[:self.lag]
        for z_i in z[self.lag:]:
            z_delagged = torch.cat([z_delagged, (z_i + z_delagged[-self.lag].detach()).unsqueeze(0)], dim=0)
        
        return z_delagged, previous_loss


class FlowLayer(LayerList):
    
    """ **FlowLayer** is the class to instantiate a layer of the flow-based model.
    
        Parameters
        ----------
            * D: dimension of the input
            * NN: neural networks used in the **AffineCoupling**
    """
    
    def __init__(self, D, NN):
        super(FlowLayer, self).__init__()
        self.D = D
        layers = []
        layers.append(ActivationNormalization(D))
        layers.append(AffineCoupling(D, NN))
        self.layers = nn.ModuleList(layers)


class FlowModel(LayerList, nn.Module):
    
    """ **FlowModel**: example of flow-based model architecture class.
    
        Parameters
        ----------
            * D: dimension of the input
            * NN: neural networks used in the **AffineCoupling**
            * depth: depth of the flow-based model (number of FlowLayers)
    """
    
    def __init__(self, D, NN, depth=1):
        super(FlowModel, self).__init__()
        self.D = D
        self.depth = depth
        
        layers = []
        for d in range(depth):
            shuffled_layer = Shuffle(D)
            layers.append(shuffled_layer)
            layers.append(ActivationNormalization(D))
            layers.append(AffineCoupling(D, NN(), volume_preservation=False))
        
        layers.append(Prior(Variable(torch.zeros(1, D)), Variable(torch.zeros(1, D))))
        self.layers = nn.ModuleList(layers)
