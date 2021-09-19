#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 11:19:12 2021

@author: lihongyi
"""
import numpy as np
import scipy as sp
from sklearn.preprocessing import StandardScaler
from data_schema import get_train_test_ml_set
import matplotlib.pyplot as plt
plt.ioff()


class NearestNeiborClassifier:
    def __init__(self, k_neighbor):
        self._k_neighbor = k_neighbor
    
    def fit(self, independent_matrix, dependent_vector):
        self._scaler = StandardScaler() 
        self._scaler.fit(independent_matrix) 
        self._independent_matrix_normal = self._scaler.transform(independent_matrix)
        
        k_neighbor = min(independent_matrix.shape[0], self._k_neighbor)
        def get_mode_of_neighbor(arr): 
            bottom_k = arr.argsort()[:k_neighbor] 
            candidates = dependent_vector[bottom_k] 
            mode = sp.stats.mode(candidates).mode[0]
            return mode
        self._get_mode_of_neighbor = get_mode_of_neighbor
    
    def predict(self, pred_matrix):
        pred_matrix_normal = self._scaler.transform(pred_matrix)
        n_tr = self._independent_matrix_normal.shape[0] 
        n_pred = pred_matrix_normal.shape[0]
        
        pred_matrix_exp = np.repeat(pred_matrix_normal, n_tr, axis=0)
        independent_matrix_exp = np.tile(self._independent_matrix_normal, (n_pred, 1))
        diff_exp = pred_matrix_exp - independent_matrix_exp
        diff_matrix = np.sqrt((diff_exp * diff_exp).sum(axis=1)).reshape((n_pred, n_tr))
        
        pred_vector = np.apply_along_axis(self._get_mode_of_neighbor, 1, diff_matrix)
        return pred_vector


def train_k_nearest_neibor(independent_matrix, dependent_vector, k_neighbor):
    clf = NearestNeiborClassifier(k_neighbor)
    clf.fit(independent_matrix, dependent_vector)
    return clf


def error_rate(independent_matrix, dependent_vector, clf):
    pred = clf.predict(independent_matrix)
    miss = pred != dependent_vector
    err = sum(miss) / len(miss)
    return err   


def cv_err_by_k_neighbor(train_set, k_neighbor, k=10):
    n = len(train_set.dependent_vector)
    
    errs_validate = []
    for i in range(k):
        idx_map = np.array([True] * n)
        if i == k-1:
            idx_map[(n // k)*i: ] = False
        else:
            idx_map[(n // k)*i: (n // k)*(i+1)] = False
        
        clf = train_k_nearest_neibor(train_set.independent_matrix[idx_map], train_set.dependent_vector[idx_map], k_neighbor)
        err_validate = error_rate(train_set.independent_matrix[~idx_map], train_set.dependent_vector[~idx_map], clf)
        errs_validate.append(err_validate)
    
    return np.mean(errs_validate)


def best_k_neighbor_by_cv(train_set, min_k_neighbor=1, max_k_neighbor=20, k=10, plot_name=None):
    ks = []
    cv_errs = []
    for kn in range(min_k_neighbor, max_k_neighbor+1):
        ks.append(kn)
        cv_errs.append(cv_err_by_k_neighbor(train_set, kn, k=k))
        print(kn)
    best_k = ks[np.argmin(np.round(cv_errs, 5))]
    
    if plot_name is not None:
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(ks, cv_errs)
        plt.axvline(x=best_k, color='red', linestyle='--')
        ax.set(xlabel='nb of unweighted neighbors', ylabel='cross-validation mean error rate', 
               title=f'{k}-fold cross validation for K-nearest-neighbor on {plot_name}')
        fig.savefig(f'{plot_name}_knn_cv.png')
        plt.clf()
        plt.close()
    return best_k


def run(data_set_name):
    train_set, test_set = get_train_test_ml_set(data_set_name)
    best_k = best_k_neighbor_by_cv(train_set, plot_name=data_set_name)
    clf = train_k_nearest_neibor(train_set.independent_matrix, train_set.dependent_vector, best_k)
    test_err = error_rate(test_set.independent_matrix, test_set.dependent_vector, clf)
    with open(f'{data_set_name}_knn_test_err.txt', 'w') as f:
        f.write('%d' % test_err)  


def main():
    run('abalone')


if __name__ == '__main__':
    main()
