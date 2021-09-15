#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 17:21:37 2021

@author: lihongyi
"""

import numpy as np
from sklearn import tree
from data_schema import get_train_test_ml_set
import matplotlib.pyplot as plt
plt.ioff()


def train_tree(independent_matrix, dependent_vector, depth):
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth)
    clf = clf.fit(independent_matrix, dependent_vector)
    return clf


def error_rate(independent_matrix, dependent_vector, clf):
    pred = clf.predict(independent_matrix)
    miss = pred != dependent_vector
    err = sum(miss) / len(miss)
    return err    


def cv_err_by_tree_depth(train_set, depth, k=10):
    n = len(train_set.dependent_vector)
    
    errs_validate = []
    for i in range(k):
        idx_map = np.array([True] * n)
        if i == k-1:
            idx_map[(n // k)*i: ] = False
        else:
            idx_map[(n // k)*i: (n // k)*(i+1)] = False
        
        clf = train_tree(train_set.independent_matrix[idx_map], train_set.dependent_vector[idx_map], depth)
        err_validate = error_rate(train_set.independent_matrix[~idx_map], train_set.dependent_vector[~idx_map], clf)
        errs_validate.append(err_validate)
    
    return np.mean(errs_validate)


def best_tree_depth_by_cv(train_set, min_depth=2, max_depth=20, k=10, plot_name=None):
    depths = []
    cv_errs = []
    for d in range(min_depth, max_depth+1):
        depths.append(d)
        cv_errs.append(cv_err_by_tree_depth(train_set, d, k=k))
    best_d = depths[np.argmin(np.round(cv_errs, 5))]
    
    if plot_name is not None:
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(depths, cv_errs)
        plt.axvline(x=best_d, color='red', linestyle='--')
        ax.set(xlabel='tree max depth', ylabel='cross-validation mean error rate', 
               title=f'{k}-fold cross validation for decision tree on {plot_name}')
        fig.savefig(f'{plot_name}_dt_cv.png')
        plt.clf()
        plt.close()
    return best_d


def run(data_set_name):
    train_set, test_set = get_train_test_ml_set(data_set_name)
    best_d = best_tree_depth_by_cv(train_set, plot_name=data_set_name)
    clf = train_tree(train_set.independent_matrix, train_set.dependent_vector, best_d)
    test_err = error_rate(test_set.independent_matrix, test_set.dependent_vector, clf)
    with open(f'{data_set_name}_dt_test_err.txt', 'w') as f:
        f.write('%d' % test_err)


def main():
    run('abalone')


if __name__ == '__main__':
    main()
