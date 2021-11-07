#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 21:44:28 2021

@author: lihongyi
"""
import numpy as np
import matplotlib.pyplot as plt
import variable_trans
import kmeans
import scipy


cov_ = np.array([[1, .0],[.0, 0]])
mean_ = np.array([0, 0])


theta1 = .6
theta2 = 1.7

rot1 = np.array([[np.cos(theta1),-np.sin(theta1)],[np.sin(theta1),np.cos(theta1)]])
rot2 = np.array([[np.cos(theta2),-np.sin(theta2)],[np.sin(theta2),np.cos(theta2)]])

sample1 = np.random.RandomState(123).multivariate_normal(mean_, cov_, size=(100,)).dot(rot1)
sample2 = np.random.RandomState(456).multivariate_normal(mean_, cov_, size=(100,)).dot(rot2)

sample = np.concatenate((sample1, sample2), axis=0)
scipy.stats.kurtosis(sample[:, 0], fisher=True)
scipy.stats.kurtosis(sample[:, 1], fisher=True)


lable = np.array([1]*100+[2]*100)
plt.scatter(sample[:, 0], sample[:, 1], c=lable, s=lable, alpha = 0.9)
plt.savefig('ic_data_true.png')
plt.close()

ica = variable_trans.perform_ica(sample, name='adhoc')

ic_sample = ica.transform(sample)
scipy.stats.kurtosis(ic_sample[:, 0], fisher=True)
scipy.stats.kurtosis(ic_sample[:, 1], fisher=True)

cluster = kmeans.fit_k_means(ic_sample, 2)
plt.scatter(sample[:, 0], sample[:, 1], c=cluster.labels_+1, s=cluster.labels_+1, alpha = 0.9)
plt.savefig('ic_data_notran.png')
plt.close()

ic_sample = variable_trans.ic_wt2(ica.transform(sample), 2)
cluster = kmeans.fit_k_means(ic_sample, 2)
plt.scatter(sample[:, 0], sample[:, 1], c=cluster.labels_+1, s=cluster.labels_+1, alpha = 0.9)
plt.savefig('ic_data_tran.png')
plt.close()
