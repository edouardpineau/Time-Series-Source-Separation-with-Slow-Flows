ATTENTION: notebook to be updated

# Time Series Source Separation with Slow Flows

The current repository offers an implementation of the paper [Time Series Source Separation with Slow Flows](https://arxiv.org/pdf/2007.10182.pdf) presented as a contribution at the [second ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models (2020)](https://invertibleworkshop.github.io/). 

### Abstract

In this paper, we show that slow feature analysis (SFA), a common time series decomposition method, naturally fits into the flow-based models (FBM) framework, a type of invertible neural latent variable models. Building upon recent advances on blind source separation, we show that such a fit makes the time series decomposition
identifiable.

### Dependencies

The current version of the code uses the module [SOBI.py](https://github.com/edouardpineau/Time-Series-ICA/blob/master/SOBI.py). It is a linear ICA specialized for sequence decomposition. I will package it soon; for now, it is duplicated in the current repository.

### Citing

    @inproceedings{pineau2020time,
                   title={Time Series Source Separation with Slow Flows},
                   author={Pineau, Edouard and Razakarivony, S{\'e}bastien and Bonald, Thomas},
                   booktitle={ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models},
                   year={2020}
    }
