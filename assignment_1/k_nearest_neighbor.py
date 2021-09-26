#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 11:19:12 2021

@author: lihongyi
"""
import numpy as np
import scipy as sp
from typing import List
from sklearn.preprocessing import StandardScaler
from data_schema import get_train_test_ml_set
import matplotlib.pyplot as plt
plt.ioff()


class NearestNeiborClassifier:
    def __init__(self, ks_neighbor: List):
        self._ks_neighbor = ks_neighbor
    
    def fit(self, independent_matrix, dependent_vector):
        self._scaler = StandardScaler() 
        self._scaler.fit(independent_matrix) 
        self._independent_matrix_normal = self._scaler.transform(independent_matrix)
        
        def get_mode_of_neighbor(arr): 
            arg_sort = arr.argsort()
            modes = []
            for k in self._ks_neighbor:
                bottom_k = arg_sort[:min(k, len(arr))] 
                candidates = dependent_vector[bottom_k] 
                modes.append(sp.stats.mode(candidates).mode[0])
            return modes
        self._get_mode_of_neighbor = get_mode_of_neighbor
    
    def predict(self, pred_matrix, slice_size=100):
        pred_matrix_normal = self._scaler.transform(pred_matrix)
        n_tr = self._independent_matrix_normal.shape[0] 
        n_pred = pred_matrix_normal.shape[0]
        
        tr_norm2 = (self._independent_matrix_normal * self._independent_matrix_normal).sum(axis=1)
        pred_norm2 = (pred_matrix_normal * pred_matrix_normal).sum(axis=1)
        inner_prod = pred_matrix_normal.dot(self._independent_matrix_normal.transpose())
        
        diff_matrix = -2 * inner_prod
        for i in range(n_pred):
            diff_matrix[i, :] += pred_norm2[i]
        for j in range(n_tr):
            diff_matrix[:, j] += tr_norm2[j]
        
        pred_vector = np.apply_along_axis(self._get_mode_of_neighbor, 1, diff_matrix).transpose()
        return pred_vector
    
    def predict_old(self, pred_matrix, slice_size=1000):
        pred_matrix_normal = self._scaler.transform(pred_matrix)
        n_pred = pred_matrix_normal.shape[0]
        
        pred_list = [[] for k in self._ks_neighbor]
        for i in range(0, n_pred, slice_size):
            stt = i
            end = min(stt + slice_size, n_pred)
            pred_slice = self.predict_slice(pred_matrix_normal[stt:end, :])
            for k, p in enumerate(pred_list):
                p.extend(pred_slice[k])
            print(i)
        
        pred_vector = np.array(pred_list)
        return pred_vector
    
    def predict_slice(self, pred_matrix_normal):
        n_tr = self._independent_matrix_normal.shape[0] 
        n_pred = pred_matrix_normal.shape[0]
        
        pred_matrix_exp = np.repeat(pred_matrix_normal, n_tr, axis=0)
        independent_matrix_exp = np.tile(self._independent_matrix_normal, (n_pred, 1))
        diff_exp = pred_matrix_exp - independent_matrix_exp
        diff_matrix = np.sqrt((diff_exp * diff_exp).sum(axis=1)).reshape((n_pred, n_tr))
        
        pred_vector = np.apply_along_axis(self._get_mode_of_neighbor, 1, diff_matrix)
        pred_list = pred_vector.transpose().tolist()
        return pred_list


def train_k_nearest_neibor(independent_matrix, dependent_vector, ks_neighbor: List):
    clf = NearestNeiborClassifier(ks_neighbor)
    clf.fit(independent_matrix, dependent_vector)
    return clf


def error_rate(independent_matrix, dependent_vector, clf, ks_neighbor: List):
    preds = clf.predict(independent_matrix)
    errs = []
    for k in ks_neighbor:
        i = clf._ks_neighbor.index(k)
        pred = preds[i] 
        miss = pred != dependent_vector 
        err = sum(miss) / len(miss)
        errs.append(err)
    return errs


def cv_err_by_k_neighbor(train_set, ks_neighbor: List, k=10):
    n = len(train_set.dependent_vector)
    
    errs_validate = []
    for i in range(k):
        idx_map = np.array([True] * n)
        if i == k-1:
            idx_map[(n // k)*i: ] = False
        else:
            idx_map[(n // k)*i: (n // k)*(i+1)] = False
        
        clf = train_k_nearest_neibor(train_set.independent_matrix[idx_map], train_set.dependent_vector[idx_map], ks_neighbor)
        err_validate = error_rate(train_set.independent_matrix[~idx_map], train_set.dependent_vector[~idx_map], clf, ks_neighbor)
        errs_validate.append(err_validate)
        print(i)
    errs_validate = np.array(errs_validate)
    
    return np.mean(errs_validate, axis=0)


def best_k_neighbor_by_cv(train_set, min_k_neighbor=1, max_k_neighbor=30, k=10, plot_name=None):
    ks = [k for k in range(min_k_neighbor, max_k_neighbor+1)]
    cv_errs = cv_err_by_k_neighbor(train_set, ks, k=k)
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
    clf = train_k_nearest_neibor(train_set.independent_matrix, train_set.dependent_vector, [best_k])
    test_err = error_rate(test_set.independent_matrix, test_set.dependent_vector, clf, [best_k])[0]
    with open(f'{data_set_name}_knn_test_err.txt', 'w') as f:
        f.write('%.5f' % test_err)  


def main():
    run('abalone')
    run('bank-additional')


if __name__ == '__main__':
    main()
