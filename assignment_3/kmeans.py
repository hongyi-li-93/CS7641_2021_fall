#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:25:43 2021

@author: lihongyi
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import gap_score

seed = 123


def fit_k_means(independent_matrix, k):
    kmeans = KMeans(n_clusters=k, random_state=seed, algorithm='full')
    kmeans.fit(independent_matrix)
    return kmeans


def get_silhouette_by_k(independent_matrix, min_k=3, max_k=20, step_size=1, name=None):
    scores = []
    clusters = []
    ks = []
    for k in range(min_k, max_k+1, step_size):
        kmeans = fit_k_means(independent_matrix, k)
        score = silhouette_score(independent_matrix, kmeans.labels_, metric='euclidean')
        scores.append(score)
        clusters.append(kmeans)
        ks.append(k)
        print(k)
    if name is not None:
        plt.plot(np.array(ks), np.array(scores)) 
        plt.xlabel('nb of clusters by kmeans') 
        plt.ylabel('silhouette score') 
        plt.savefig(f'{name}_kmeans_silhouette.png')
        plt.close()
    return scores, clusters


def get_gap_by_k(independent_matrix, min_k=3, max_k=20, step_size=1, name=None):
    gs = []
    ss = []
    ks = []
    clusters = []
    for k in range(min_k, max_k+1, step_size):
        kmeans = fit_k_means(independent_matrix, k)
        w = gap_score.get_w(kmeans.labels_, independent_matrix)
        ref_samples = gap_score.get_reference_samples(independent_matrix)
        ref_lables = [fit_k_means(r, k).labels_ for r in ref_samples]
        assert len(ref_lables) == len(ref_samples)
        w_s = [gap_score.get_w(lab, r) for lab, r in zip(ref_lables, ref_samples)]
        l, s = gap_score.get_l_s(w_s)
        g = l - np.log(w)
        
        gs.append(g)
        ss.append(s)
        ks.append(k)
        clusters.append(kmeans)
        print(k)
    if name is not None:
        plt.plot(np.array(ks), np.array(gs), label='gap', linestyle="-") 
        plt.plot(np.array(ks), np.array(gs) - np.array(ss), label='gap - s', linestyle="-.") 
        plt.plot(np.array(ks), np.array(gs) + np.array(ss), label='gap + s', linestyle="-.") 
        plt.xlabel('nb of clusters by kmeans') 
        plt.ylabel('gap score') 
        plt.legend()
        plt.savefig(f'{name}_kmeans_gap.png')
        plt.close()
    return gs, ss, clusters
    