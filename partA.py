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
from sklearn.metrics import precision_recall_fscore_support
import time

start_time = time.time()


def class_estimate(sample, mu, cov):
    np_sample = np.array(sample).reshape(sample.__len__(), 1)
    np_mu = np.array(mu).reshape(sample.__len__(), 1)
    part = (np_sample-np_mu)
    full = np.matmul(-.5*np.matmul(part.transpose(), cov), part)  # [0][0]
    print(full)
    return full[0][0]


# V00746112
classValue1 = 1
classValue2 = 2
percTrain = .01
percTest = .2
(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.set_printoptions(linewidth=250)

x_train_mine, y_train_mine = extractMine(
    x_train, y_train,  classValue1, classValue2)
numToTrainOn = int(percTrain*x_train_mine.__len__())

x_train_mine = x_train_mine[:numToTrainOn]
y_train_mine = y_train_mine[:numToTrainOn]

# y_train_mine = np.array(y_train_mine).reshape(y_train_mine.__len__(), 1)
# print(x_train_mine[0])
# print()
print(x_train_mine[0])
x_train_mine_selected = featureSelection(x_train_mine)
print(x_train_mine_selected[0])

# print(x_train_mine_selected[0])
# normalize data and swap the grayscale 1-255 value for a 1 only
x_train_mine_norm = []
for i in range(0, x_train_mine_selected.__len__()):
    x_train_mine_norm.append((x_train_mine_selected[i]))
    # x_train_mine_norm.append(normalize(x_train_mine_selected[i]))

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
cov_est = map_estimate1['estimated_cov']
mu0_est = map_estimate1['estimated_mu0']
mu1_est = map_estimate1['estimated_mu1']
y_est = []

x_test_mine, y_test_mine = extractMine(
    x_train, y_train, classValue1, classValue2)
numToTestOn = int(percTest*x_test_mine.__len__())

x_test_mine = x_test_mine[:numToTestOn]
y_test_mine = y_test_mine[:numToTestOn]

x_test_mine_class1 = extractClass(x_test_mine, y_test_mine, 1)
x_test_mine_class2 = extractClass(x_test_mine, y_test_mine, 0)

x_test_mine_selected = featureSelection(x_test_mine)
x_test_mine_norm = []
for i in range(0, x_test_mine_selected.__len__()):
    x_test_mine_norm.append((x_test_mine_selected[i]))
    # x_test_mine_norm.append(normalize(x_test_mine_selected[i]))

# sample some random point in 2D feature space
x_test_mine_norm_flat = flatten(x_test_mine_norm)
numClass1 = 0
numClass2 = 0
y_ests = []
inv_cov= inv(np.array(cov_est))

for i in range(0, x_test_mine_norm_flat.__len__()):
    if class_estimate(x_test_mine_norm_flat[i], mu0_est, inv_cov) > class_estimate(x_test_mine_norm_flat[i], mu1_est, inv_cov):
        y_ests.append(classValue2)
        numClass2 += 1
    else:
        y_ests.append(classValue1)
        numClass1 += 1


print("That took", time.time()-start_time, "seconds to run")
print("We predicted we have", numClass1, "images of", classValue1, "'s.  We actually have",
      x_test_mine_class1.__len__(), "images of", classValue1, "'s")
print("We predicted we have", numClass2, "images of", classValue2, "'s.  We actually have",
      x_test_mine_class2.__len__(), "images of", classValue2, "'s")

print("Accuracy is", computeAccuracy(y_test_mine, y_ests)*100, "% using",
      y_train_mine.__len__(), "training samples and", y_test_mine.__len__(), "testing samples, each with", numberOfFeatures, "features.")

# print([classValue2 if x==0 else x for x in y_test_mine])
# print(y_ests)
# print(precision_recall_fscore_support([classValue2 if x==0 else x for x in y_test_mine], y_ests, labels=[0, 1]))
# print(y_test_mine)
# print(y_ests)
# compare map_estimate1['estimated_mu1'] with true_mu1
# same for mu_2, cov
