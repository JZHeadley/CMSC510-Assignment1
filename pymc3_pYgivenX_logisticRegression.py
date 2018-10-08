#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tarodz
"""

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import theano.tensor as T

### START OF SOME FAKE DATASET GENERATION
np.random.seed(123)

# number of samples in total
sCnt = 100;
numberOfFeatures = 2;

# true parameters w and b
true_w = np.zeros((numberOfFeatures, 1));
true_w[0] = -0.5;
true_w[1] = 1.3;
true_b = -0.3;

# sample some random point in 2D feature space
X = np.random.randn(sCnt, numberOfFeatures).astype(dtype='float32');

# calculate u=w^Tx+b
# true_u = true_w1*X[:,0] + true_w2*X[:,1] + true_b;
true_u = np.transpose(true_w) * X + true_b;

# P(+1|x)=a(u) #see slides for def. of a(u)
probPlusOne = 1.0 / (1.0 + np.exp(-1.0 * true_u));

# sample realistic (i.e. based on pPlusOne, but not deterministic) class values for the dataset
# class +1 is comes from a probability distribution with probability "prob" for +1, and 1-prob for class 0
Y = np.random.binomial(1, probPlusOne);

#### END OF FAKE DATASET GENERATION

##### for MNIST experiment:
# replace the above with x=.... so that x is the [samples x features] matrix from MNIST dataset.
# replace the above with vector Y of actual classes of samples (0: class A, 1: class B) from the MNIST dataset.


#### START OF MODEL BUILDING AND ESTIMATION

# instantiate an empty PyMC3 model
basic_model = pm.Model()

# fill the model with details:
with basic_model:
    mu_prior_cov = 100 * np.eye(numberOfFeatures);
    mu_prior_mu = np.zeros((numberOfFeatures,))

    # Priors for w,b (Gaussian priors), centered at 0, with very large std.dev.
    w = pm.MvNormal('estimated_w', mu=mu_prior_mu, cov=mu_prior_cov, shape=numberOfFeatures)
    b = pm.Normal('estimated_b', 0, 100);

    # calculate u=w^Tx+b
    # here w, b are unknown to be estimated from data
    # X is the known data matrix [samples x features]
    u = w * X + b;
    # P(+1|x)=a(u) #see slides for def. of a(u)
    prob = 1.0 / (1.0 + T.exp(-1.0 * u));

    # class +1 is comes from a probability distribution with probability "prob" for +1, and 1-prob for class 0
    # here Y is the known vector of classes
    # prob is (indirectly coming from the estimate of w,b and the data x)
    Y_obs = pm.Bernoulli('Y_obs', p=prob, observed=Y);

# done with setting up the model


# now perform maximum likelihood (actually, maximum a posteriori (MAP), since we have priors) estimation
# map_estimate1 is a dictionary: "parameter name" -> "it's estimated value"
map_estimate1 = pm.find_MAP(model=basic_model)

# we can also do MCMC sampling from the distribution over the parameters
# and e.g. get confidence intervals
with basic_model:
    # obtain starting values via MAP
    start = pm.find_MAP()

    # instantiate sampler
    step = pm.Slice()

    # draw 10000 posterior samples
    # can take rather long time
    trace = pm.sample(10000, step=step, start=start)

pm.traceplot(trace)
pm.summary(trace)
plt.show()