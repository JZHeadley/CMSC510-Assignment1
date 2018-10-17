#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tarodz
Partially adapted from: http://austinrochford.com/posts/2015-09-16-mvn-pymc3-lkj.html
"""
from keras.datasets import mnist

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import theano.tensor as T
from support import *
from numpy.linalg import inv


def class_estimate(sample, mu, cov):
    return ((-.5*np.array(sample-mu).transpose())*inv(np.array(cov))*np.array(sample)-np.array(mu))


# V00746112
classValue1 = 1
classValue2 = 2
numToTrainOn = 50
numToTestOn = 100
(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.set_printoptions(linewidth=250)

x_train_mine, y_train_mine = extractMine(
    x_train, y_train,  classValue1, classValue2)

x_train_mine = x_train_mine[:numToTrainOn]
y_train_mine = y_train_mine[:numToTrainOn]

# y_train_mine = np.array(y_train_mine).reshape(y_train_mine.__len__(), 1)
# print(x_train_mine[0])
# print()
x_train_mine_selected = featureSelection(x_train_mine)
# print(x_train_mine_selected[0])
# normalize data and swap the grayscale 1-255 value for a 1 only
x_train_mine_norm = []
for i in range(0, x_train_mine_selected.__len__()):
    x_train_mine_norm.append(normalize(x_train_mine_selected[i]))

# sample some random point in 2D feature space
x_train_mine_norm_flat = flatten(x_train_mine_norm)
numberOfFeatures = x_train_mine_norm_flat[0].__len__()

print(numberOfFeatures, "features")


class1 = extractClass(x_train_mine_norm_flat, y_train_mine, classValue1)
class2 = extractClass(x_train_mine_norm_flat, y_train_mine, 0)

# print(class1[0])
x1 = np.array(class1)
x0 = np.array(class2)
# print(x1[0])

# replace the above with vectors from the MNIST dataset, x1 for class A, x0 for class B


# END OF FAKE DATASET GENERATION

# for MNIST experiment:
# replace the above with two matrices, x0 and x1 [samples x features] from the MNIST dataset, x1 for class A, x0 for class B


# START OF MODEL BUILDING AND ESTIMATION

# instantiate an empty PyMC3 model
basic_model = pm.Model()

# fill the model with details:
with basic_model:
    # parameters for priors for gaussian means
    mu_prior_cov = 100 * np.eye(numberOfFeatures)
    mu_prior_mu = np.zeros((numberOfFeatures,))

    # Priors for gaussian means (Gaussian prior): mu1 ~ N(mu_prior_mu, mu_prior_cov), mu0 ~ N(mu_prior_mu, mu_prior_cov)
    mu1 = pm.MvNormal('estimated_mu1', mu=mu_prior_mu,
                      cov=mu_prior_cov, shape=numberOfFeatures)
    mu0 = pm.MvNormal('estimated_mu0', mu=mu_prior_mu,
                      cov=mu_prior_cov, shape=numberOfFeatures)

    # Prior for gaussian covariance matrix (LKJ prior):
    # see here for details: http://austinrochford.com/posts/2015-09-16-mvn-pymc3-lkj.html
    # and here: http://docs.pymc.io/notebooks/LKJ.html
    sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=numberOfFeatures)
    chol_packed = pm.LKJCholeskyCov('chol_packed',
                                    n=numberOfFeatures, eta=2, sd_dist=sd_dist)
    chol = pm.expand_packed_triangular(numberOfFeatures, chol_packed)
    cov_mx = pm.Deterministic('estimated_cov', chol.dot(chol.T))

    # observations x1, x0 are supposed to be P(x|y=class1)=N(mu1,cov_both), P(x|y=class0)=N(mu0,cov_both)
    # here is where the Dataset (x1,x0) comes to influence the choice of paramters (mu1,mu0, cov_both)
    # this is done through the "observed = ..." argument; note that above we didn't have that
    x1_obs = pm.MvNormal('x1', mu=mu1, chol=chol, observed=x1)
    x0_obs = pm.MvNormal('x0', mu=mu0, chol=chol, observed=x0)

# done with setting up the model


# now perform maximum likelihood (actually, maximum a posteriori (MAP), since we have priors) estimation
# map_estimate1 is a dictionary: "parameter name" -> "it's estimated value"
map_estimate1 = pm.find_MAP(model=basic_model)
print("estimate is", map_estimate1)
cov_est = map_estimate1['estimated_cov']
mu0_est = map_estimate1['estimated_mu0']
mu1_est = map_estimate1['estimated_mu1']
print("cov", cov_est)
print("mu0", mu0_est)
print("mu1", mu1_est)
class_est = []

x_test_mine, y_test_mine = extractMine(
    x_test, y_test, classValue1, classValue2)

x_test_mine = x_test_mine[:numToTestOn]
y_test_mine = y_test_mine[:numToTestOn]

x_test_mine_selected = featureSelection(x_test_mine)
x_test_mine_norm = []
for i in range(0, x_test_mine_selected.__len__()):
    x_test_mine_norm.append(normalize(x_test_mine_selected[i]))

# sample some random point in 2D feature space
x_test_mine_norm_flat = flatten(x_test_mine_norm)
class_est = []
print("prob class 2")
print(class_estimate(x_test_mine_norm_flat[0], mu0_est, cov_est))
print("prob class 1")
print(class_estimate(x_test_mine_norm_flat[0], mu1_est, cov_est))
for i in range(0, x_test_mine_norm_flat.__len__()):
    if class_estimate(x_test_mine_norm_flat[i], mu0_est, cov_est) > class_estimate(x_test_mine_norm_flat[i], mu1_est, cov_est):
        class_est[i] = class2
    else:
        class_est[i] = class1

print(class_est)
# compare map_estimate1['estimated_mu1'] with true_mu1
# same for mu_2, cov
