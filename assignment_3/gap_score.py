#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:06:50 2021

@author: lihongyi

I am going to implement GAP score here (for L2 distance only)
"""

import numpy as np


def get_sample_l2(samples):
    l2 = (samples * samples).sum(axis=1)
    return l2


def get_where_claster(lables, cluster):
    where = np.where(lables == cluster)
    n_s = len(where[0])
    return where, n_s


def get_d_cluster(where, samples, sample_l2):
    n_s = len(where[0])
    s_samples = samples[where]
    mu = s_samples.mean(axis=0)
    mu_l2 = mu.dot(mu.T)
    s_sample_l2 = sample_l2[where]
    sum_l2 = s_sample_l2.sum() - n_s * mu_l2
    return 2 * n_s * sum_l2


def get_w(lables, samples):
    assert lables.shape[0] == samples.shape[0]
    sample_l2 = get_sample_l2(samples)
    
    w = 0
    lbs = np.unique(lables)
    for cluster in lbs:
        where, n_s = get_where_claster(lables, cluster)
        w += get_d_cluster(where, samples, sample_l2) / (2 * n_s)
    return w


def get_reference_samples(samples, nb=50, seed=123):
    m, n = samples.shape
    stacked_reference = np.zeros(shape=(m * nb, n))
    
    np.random.seed(seed)
    for fi in range(n):
        stacked_reference[:, fi] += np.random.uniform(low=samples[:, fi].min(), high=samples[:, fi].max(), size=(m * nb,))
    
    refs = []
    for ri in range(nb):
        refs.append(stacked_reference[ri * m: (ri+1) * m, :])
    return refs


def get_l_s(w_s):
    nb = len(w_s)
    l = 0
    l2 = 0
    for w in w_s:
        lw = np.log(w)
        l += lw / nb
        l2 += (lw * lw) / nb
    
    sd = np.sqrt(l2 - l*l)
    s = sd * np.sqrt(1 + 1/nb)
    return l, s

