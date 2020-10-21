# Time Series Source Separation with Slow Flows

The current repository offers an implementation of the paper [Time Series Source Separation with Slow Flows](https://arxiv.org/pdf/2007.10182.pdf) presented as a contribution at the [second ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models (2020)](https://invertibleworkshop.github.io/). 

### Abstract

In this paper, we show that slow feature analysis (SFA), a common time series decomposition method, naturally fits into the flow-based models (FBM) framework, a type of invertible neural latent variable models. Building upon recent advances on blind source separation, we show that such a fit makes the time series decomposition
identifiable.

### Introduction

Decomposing data into independent components sometimes is not enough to find relevant information: we need to find the *source factors* from which data have been generated. We not **X** the data, **S** the factors and **f(S)=X** the *unknown invertible* mixing. While in the linear case (**f=A**), independent component analysis (ICA) identifies the *true sources* (up to scaling and rotation) [1], the non-linear case has a major issue: there exists an infinite number of solutions. Recent works have proposed new proofs of identifiability in the non-linear case under three main assumptions: universal approximation function to estimate f, infinite data and access to additional information about the data from which we can extract a relevant *inductive bias* [2, 3, 4]. In particular, they use the recent advances in data representation learning with neural networks (universal approximation functions that scale on large datasets). 

In this paper, we couple a known time series representation inductive bias called *slowness* to the recent neural network based non-linear identifiable ICA. 

### Slowness

*Slowness* is a common temporal structure used in time series decomposition. It represents the fact that two consecutive time-steps in a time series have close values. It is a common assumption that relevant factors underlying data are slower than their mixing [5]. If we note **(f,Z)** the estimated mixing function and factors, then the slowness is defined as

<p align="center">
  <img src="https://github.com/edouardpineau/Time-Series-Source-Separation-with-Slow-Flows/blob/main/images/eq_slowness.png" width="350">
</p>

with <img src="https://latex.codecogs.com/gif.latex?\Delta  " /> the temporal differenciation operator. 

In [6] they redefine slowness in term of maximum-likelihood, i.e.

<p align="center">
  <img src="https://github.com/edouardpineau/Time-Series-Source-Separation-with-Slow-Flows/blob/main/images/eq_ML_slowness_2.png" width="250">
</p>

<p align="center">
    <img src="https://github.com/edouardpineau/Time-Series-Source-Separation-with-Slow-Flows/blob/main/images/eq_ML_slowness.png" width="220">
</p>

Finally, we use recent ICA methods that use invertible neural networks trained by maximum-likelihood [7, 8] to decompose data. They are called *flow-based models* (FBMs). 

Our paper combines these concepts: introducing slowness into FBMs (*slow-FBMs*) proves identifiability of the sources up-to scaling and rotation. 

##### Structural time series

<p align="center">
  <img src="https://github.com/edouardpineau/Time-Series-Source-Separation-with-Slow-Flows/blob/main/images/slowness_sound.png" width="350">
</p>


<p align="center">
  <img src="https://github.com/edouardpineau/Time-Series-Source-Separation-with-Slow-Flows/blob/main/images/slowness_sound_table.png" width="350">
</p>

##### Sound mixing

<p align="center">
  <img src="https://github.com/edouardpineau/Time-Series-Source-Separation-with-Slow-Flows/blob/main/images/slowness_structural.png" width="350">
</p>

<p align="center">
  <img src="https://github.com/edouardpineau/Time-Series-Source-Separation-with-Slow-Flows/blob/main/images/slowness_structural_table.png" width="350">
</p>

### Citing

    @inproceedings{pineau2020time,
                   title={Time Series Source Separation with Slow Flows},
                   author={Pineau, Edouard and Razakarivony, S{\'e}bastien and Bonald, Thomas},
                   booktitle={ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models},
                   year={2020}
    }
