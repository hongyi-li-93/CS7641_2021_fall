#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:57:32 2021

@author: lihongyi
"""
import numpy as np
import matplotlib.pyplot as plt
import kgauss
import kmeans


cov1 = np.array([[1, .95],[.95, 1]])
cov2 = np.array([[1, -.85],[-.85, 1]])
mean_ = np.array([0, 0])

sample1 = np.random.RandomState(123).multivariate_normal(mean_, cov1, size=(100,))
sample2 = np.random.RandomState(456).multivariate_normal(mean_, cov2, size=(100,))

sample = np.concatenate((sample1, sample2), axis=0)
lable = np.array([1]*100+[2]*100)
plt.scatter(sample[:, 0], sample[:, 1], c=lable, s=lable, alpha = 0.9)
plt.savefig('gauss_data_true.png')
plt.close()


cluster = kgauss.fit_k_gaussians(sample, 2)
plt.scatter(sample[:, 0], sample[:, 1], c=cluster.predict(sample)+1, s=cluster.predict(sample)+1, alpha = 0.9)
plt.savefig('gauss_data_kgauss.png')
plt.close()


cluster = kmeans.fit_k_means(sample, 2)
plt.scatter(sample[:, 0], sample[:, 1], c=cluster.predict(sample)+1, s=cluster.predict(sample)+1, alpha = 0.9)
plt.savefig('gauss_data_kmeans.png')
plt.close()
