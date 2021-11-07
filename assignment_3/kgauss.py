#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 19:01:56 2021

@author: lihongyi
"""

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from data_schema import get_train_test_ml_set, plot_cluster, get_scaler
import matplotlib.pyplot as plt
import numpy as np
import gap_score

seed = 123


def fit_k_gaussians(independent_matrix, k):
    kgauss = GaussianMixture(n_components=k, 
                             covariance_type='full', 
                             init_params='random',
                             random_state=seed)
    kgauss.fit(independent_matrix)
    return kgauss


def get_silhouette_by_k(independent_matrix, min_k=3, max_k=20, step_size=1, name=None):
    scores = []
    clusters = []
    ks = []
    for k in range(min_k, max_k+1, step_size):
        kgauss = fit_k_gaussians(independent_matrix, k)
        score = silhouette_score(independent_matrix, kgauss.predict(independent_matrix), metric='euclidean')
        scores.append(score)
        clusters.append(kgauss)
        ks.append(k)
        print(k)
    if name is not None:
        plt.plot(np.array(ks), np.array(scores)) 
        plt.xlabel('nb of clusters by kmeans') 
        plt.ylabel('silhouette score') 
        plt.savefig(f'{name}_kgauss_silhouette.png')
        plt.close()
    return scores, clusters


def get_gap_by_k(independent_matrix, min_k=3, max_k=20, step_size=1, name=None):
    gs = []
    ss = []
    ks = []
    clusters = []
    for k in range(min_k, max_k+1, step_size):
        kgauss = fit_k_gaussians(independent_matrix, k)
        w = gap_score.get_w(kgauss.predict(independent_matrix), independent_matrix)
        ref_samples = gap_score.get_reference_samples(independent_matrix)
        ref_lables = [fit_k_gaussians(r, k).predict(r) for r in ref_samples]
        assert len(ref_lables) == len(ref_samples)
        w_s = [gap_score.get_w(lab, r) for lab, r in zip(ref_lables, ref_samples)]
        l, s = gap_score.get_l_s(w_s)
        g = l - np.log(w)
        
        gs.append(g)
        ss.append(s)
        ks.append(k)
        clusters.append(kgauss)
        print(k)
    if name is not None:
        plt.plot(np.array(ks), np.array(gs), label='gap', linestyle="-") 
        plt.plot(np.array(ks), np.array(gs) - np.array(ss), label='gap - s', linestyle="-.") 
        plt.plot(np.array(ks), np.array(gs) + np.array(ss), label='gap + s', linestyle="-.") 
        plt.xlabel('nb of clusters by kmeans') 
        plt.ylabel('gap score') 
        plt.legend()
        plt.savefig(f'{name}_kgauss_gap.png')
        plt.close()
    return gs, ss, clusters
