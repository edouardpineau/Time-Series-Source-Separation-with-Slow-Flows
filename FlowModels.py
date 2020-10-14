import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as wn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.autograd import Variable

import math
import numpy as np


class Layer(nn.Module):
    
    """
    La class **Layer** est une classe abstraite. Pour rappel une classe abstraite est utilisée lors de l'héritage pour forcer les classes filles à implémener certaines méthodes. Cela permet de forcer une interface. 

    On veut que tous les modules utilisés dans la construction de nos modèles soit de la forme **Layer**, soit avec:

    - Une méthode **to_embedding** pour passer de la donnée à une représentation latente
    - Une méthode **from_embedding** pour implémenter la tâche à effectuer à partir de l'embedding

    Cette classe définit un objet de base du modèle, de telle sorte que plusieurs de ces objets peuvent être combinés.
    """
    
    def __init__(self):
        super(Layer, self).__init__()
        
    def to_embedding(self, z, previous_loss):
        raise NotImplementedError
    
    def from_embedding(self, x, previous_loss):
        raise NotImplementedError


class LayerList(Layer, nn.Module):
    
    """ 
    Chaque objet **Layer** est utilisé dans une classe plus **LayerList**. 
    Cette liste a par définition au moins les mêmes méthodes que la classes abstraite **Layer**. 
    On remarquera que les méthodes *to_embedding* et *from_embedding* ont des inputs et ouputs de mêmes format. 
    Ainsi on pourra empiler les objets **Layers** comme les couches d'un réseau de neurones. 
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
    def __init__(self, D, moving_average=False, rho=0.99):
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
    def __init__(self, D, NN, volume_preservation=True):
        super(AffineCoupling, self).__init__()
        self.NN = NN
        self.D = D
        self.volume_preservation = volume_preservation
        
    def to_embedding(self, x, previous_loss):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        t = h[:, :self.D//2]
        s = torch.sigmoid(h[:, self.D//2:])
        
        if self.volume_preservation:
            s = s.pow(1 / (torch.sum(torch.log(s), dim=1, keepdim=True) + 1e-5))
        z2_ = z2 * s + t
        previous_loss += torch.sum(torch.log(s), dim=1)
        return torch.cat([z1, z2_], dim=1), previous_loss
    
    def from_embedding(self, x, previous_loss):
        z1, z2 = torch.chunk(x, 2, dim=1)
        h = self.NN(z1)
        t = h[:, :self.D//2]
        s = torch.sigmoid(h[:, self.D//2:])
        
        if self.volume_preservation:
            s = s.pow(1 / (torch.sum(torch.log(s), dim=1, keepdim=True) + 1e-5))
            
        z2_ = (z2 - t) / s
        previous_loss -= torch.sum(torch.log(s), dim=1)
        return torch.cat([z1, z2_], dim=1), previous_loss


class Linear(Layer):
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
    def __init__(self, D, NN):
        super(FlowLayer, self).__init__()
        self.D = D
        layers = []
        layers.append(ActivationNormalization(D))
        layers.append(AffineCoupling(D, NN))
        self.layers = nn.ModuleList(layers)


class FlowModel(LayerList, nn.Module):
    def __init__(self, D, NN, depth=1):
        super(FlowModel, self).__init__()
        self.D = D
        self.depth = depth
        
        layers = []
        for d in range(depth):
            shuffled_layer = Shuffle(D)
            layers.append(shuffled_layer)
            layers.append(FlowLayer(D, NN()))
            layers.append(shuffled_layer)
            layers.append(FlowLayer(D, NN()))
        
        layers.append(Linear(D))
        # layers.append(Prior(Variable(torch.zeros(1, D)), Variable(torch.zeros(1, D))))
        self.layers = nn.ModuleList(layers)