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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time

start_time = time.time()


def class_estimate(sample, mu, cov):
    np_sample = np.array(sample).reshape(sample.__len__(), 1)
    np_mu = np.array(mu).reshape(sample.__len__(), 1)
    part = (np_sample-np_mu)
    full = np.matmul(-.5*np.matmul(part.transpose(), cov), part)
    return full[0][0]


def build_model(x0, x1, numberOfFeatures):
    basic_model = pm.Model()
    with basic_model:
        mu_prior_cov = 100 * np.eye(numberOfFeatures)
        mu_prior_mu = np.zeros((numberOfFeatures,))

        mu1 = pm.MvNormal('estimated_mu1', mu=mu_prior_mu,
                          cov=mu_prior_cov, shape=numberOfFeatures)
        mu0 = pm.MvNormal('estimated_mu0', mu=mu_prior_mu,
                          cov=mu_prior_cov, shape=numberOfFeatures)
        sd_dist = pm.HalfCauchy.dist(beta=2.5, shape=numberOfFeatures)
        chol_packed = pm.LKJCholeskyCov('chol_packed',
                                        n=numberOfFeatures, eta=2, sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(numberOfFeatures, chol_packed)
        cov_mx = pm.Deterministic('estimated_cov', chol.dot(chol.T))

        x1_obs = pm.MvNormal('x1', mu=mu1, chol=chol, observed=x1)
        x0_obs = pm.MvNormal('x0', mu=mu0, chol=chol, observed=x0)

    map_estimate1 = pm.find_MAP(model=basic_model)
    cov_est = map_estimate1['estimated_cov']
    mu0_est = map_estimate1['estimated_mu0']
    mu1_est = map_estimate1['estimated_mu1']
    return (mu0_est, mu1_est, cov_est)


# V00746112
classValue1 = 1
classValue2 = 2
percTrain = 1
percTest = 1
(x_train, y_train), (x_test, y_test) = mnist.load_data()

np.set_printoptions(linewidth=250)

x_train, y_train = extractMine(
    x_train, y_train,  classValue1, classValue2)
x_test, y_test = extractMine(
    x_test, y_test, classValue1, classValue2)

numToTrainOn = int(percTrain*x_train.__len__())
numToTestOn = int(percTest*x_test.__len__())

x_train = x_train[:numToTrainOn]
y_train = y_train[:numToTrainOn]
x_test = x_test[:numToTestOn]
y_test = y_test[:numToTestOn]

x_test_class1 = extractClass(x_test, y_test, 1)
x_test_class2 = extractClass(x_test, y_test, 0)

x_train = flat_norm(x_train)
x_test = flat_norm(x_test)


featuresToKeep = featureSelection(x_train)
x_train = removeFeatures(x_train, featuresToKeep)
x_test = removeFeatures(x_test, featuresToKeep)
numberOfFeatures = x_train[0].__len__()

print(numberOfFeatures, "features")


class1 = extractClass(x_train, y_train, 1)
class2 = extractClass(x_train, y_train, 0)

x1 = np.array(class1)
x0 = np.array(class2)

mu0_est, mu1_est, cov_est = build_model(x0, x1, numberOfFeatures)

inv_cov = inv(cov_est)


numClass1 = 0
numClass2 = 0
y_ests = []
for i in range(0, x_test.__len__()):
    if class_estimate(x_test[i], mu0_est, inv_cov) > class_estimate(x_test[i], mu1_est, inv_cov):
        y_ests.append(classValue2)
        numClass2 += 1
    else:
        y_ests.append(classValue1)
        numClass1 += 1
# have to swap back to class values because I do everything as 0' and 1's
y_test = [classValue2 if x == 0 else classValue1 for x in y_test]
class1TrueCounts = extractClass(x_test, y_test, classValue1).__len__()
class2TrueCounts = extractClass(x_test, y_test, classValue2).__len__()

print("That took", time.time()-start_time, "seconds to run")
print("We predicted we have", numClass1, "images of", classValue1, "'s.  We actually have",
      class1TrueCounts, "images of", classValue1, "'s")
print("We predicted we have", numClass2, "images of", classValue2, "'s.  We actually have",
      class2TrueCounts, "images of", classValue2, "'s")

print("Accuracy is", computeAccuracy(y_test, y_ests)*100, "% using",
      y_train.__len__(), "training samples and", y_test.__len__(), "testing samples, each with", numberOfFeatures, "features.")

print(precision_recall_fscore_support(y_test, y_ests, labels=[classValue1, classValue2]))
