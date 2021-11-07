#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 19:42:42 2021

@author: lihongyi
"""

from sklearn.decomposition import PCA, FastICA, KernelPCA
import numpy as np
import matplotlib.pyplot as plt
import collections
import scipy

seed = 123


def perform_pca(independent_matrix, whiten=False, name=None):
    pca = PCA(copy=True, whiten=whiten, svd_solver='auto')
    pca.fit(independent_matrix)
    if name is not None:
        plt.plot(np.arange(pca.n_features_) + 1, pca.explained_variance_ratio_)
        plt.xlabel('nb of components') 
        plt.ylabel('pct of variance explained') 
        plt.savefig(f'{name}_pca.png')
        plt.close()
    return pca


def perform_ica(independent_matrix, whiten=False, name=None, diam=19.5):
    ica = FastICA(whiten=whiten, max_iter=50000, fun='logcosh', random_state=seed)
    ica.fit(independent_matrix)
    if name is not None:
        ics = ica.transform(independent_matrix)
        with open(f'{name}_ica.txt', 'w') as f:
            for c in range(ics.shape[1]):
                f.write('%.5f' % ((ics[:, c] ** 3).mean())**2)
                f.write('  ')
                f.write('%.5f' % scipy.stats.kurtosis(ics[:, c], fisher=True))
                f.write('\n')
        fig, axs = plt.subplots(independent_matrix.shape[1], ics.shape[1])
        fig.set_size_inches(diam, diam)
        for c in range(ics.shape[1]):
            x = ics[:, c]
            ics0 = ics * 0
            ics0[:, c] = x
            reverse_matrix = ica.inverse_transform(ics0)
            for r in range(independent_matrix.shape[1]):
                yi = reverse_matrix[:, r]
                yc = independent_matrix[:, r]
                axs[r, c].scatter(x, yi, c='b', alpha=0.4)
                axs[r, c].scatter(x, yc, c='r', alpha=0.2)
                axs[r, c].set_ylim([min(yc)-.2, max(yc)+.2])
                axs[r, c].set(xlabel=f'ic {c}', ylabel=f'f {r}')
        for ax in axs.flat:
            ax.label_outer()
        plt.savefig(f'{name}_ica.png')
        plt.close()
    return ica


def ic_wt2(ic_matrix, p):
    ic_mean = ic_matrix.mean(axis=0)
    ic_std = ic_matrix.std(axis=0)
    ic_norm_abs = np.abs((ic_matrix - ic_mean) / ic_std)
    ic_norm_abs2 = ic_norm_abs ** p
    ic_abs_sum = ic_norm_abs2.sum(axis=1)
    ic_wt = ic_norm_abs2.T / ic_abs_sum
    return ic_wt.T


def perform_kernel_pca(independent_matrix, n_components, kernel='rbf', name=None, **kernel_para_kwargs):
    kca = KernelPCA(n_components=n_components, kernel=kernel, eigen_solver='dense', fit_inverse_transform=True, **kernel_para_kwargs)
    kca.fit(independent_matrix)
    if name is not None:
        plt.plot(np.arange(n_components) + 1, kca.lambdas_)
        plt.xlabel('nb of components') 
        plt.ylabel('eigenvalue') 
        plt.savefig(f'{name}_kca.png')
        plt.close()
    return kca


class RCA():
    def __init__(self, n_components, seed=seed):
        self.n_components_ = n_components
        self.seed_ = seed
        self.tol = 0.0001
        
    def fit(self, x):
        n_obs, n_features = x.shape
        self.n_features_ = n_features
        self.rand_matrix_ = np.random.RandomState(self.seed_).uniform(size = (n_features, self.n_components_), low = -1, high = 1)
        self.y_ = x.dot(self.rand_matrix_)
        self.x_ = x.copy()
    
    def transform(self, x):
        return x.dot(self.rand_matrix_)
    
    def inverse_transform(self, ny, ids):
        y = self.y_[:, ids] if ids is not None else self.y_
        pseudo_inv_ = np.linalg.inv(y.transpose().dot(y)).dot(y.transpose().dot(self.x_))
        return ny.dot(pseudo_inv_)


def perform_rca(independent_matrix, n_components, name=None, diam=19.5):
    rca = RCA(n_components)
    rca.fit(independent_matrix)
    if name is not None:
        rcs = rca.transform(independent_matrix)
        fig, axs = plt.subplots(independent_matrix.shape[1], rcs.shape[1])
        fig.set_size_inches(diam, diam)
        for c in range(rcs.shape[1]):
            x = rcs[:, c]
            reverse_matrix = rca.inverse_transform(rcs[:, [c,]], [c,])
            for r in range(independent_matrix.shape[1]):
                yi = reverse_matrix[:, r]
                yc = independent_matrix[:, r]
                axs[r, c].scatter(x, yi, c='b', alpha=0.4)
                axs[r, c].scatter(x, yc, c='r', alpha=0.2)
                axs[r, c].set_ylim([min(yc)-.2, max(yc)+.2])
                axs[r, c].set(xlabel=f'rc {c}', ylabel=f'f {r}')
        for ax in axs.flat:
            ax.label_outer()
        plt.savefig(f'{name}_rca.png')
        plt.close()
    return rca        


def avg_entropy(label_parent, label_child, name=None):
    children = collections.Counter(label_child)
    n = len(label_child)
    e = 0
    t = 0
    for c, k in children.items():
        wt = (k + .00) / n
        t += wt * np.log(wt)
        c_parents = label_parent[np.where(label_child==c)]
        l = len(c_parents)
        parents = collections.Counter(c_parents)
        for _, p in parents.items():
            e += wt * ((p/l) * np.log(p/l))
    ae = t+e
    if name is not None:
        with open(f'{name}_avg_entropy.txt', 'w') as f:
            f.write('%.5f' % ae)
            f.write('%.5f' % t)
    return e
