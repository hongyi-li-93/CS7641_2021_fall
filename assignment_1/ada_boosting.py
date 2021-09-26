#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 18:35:36 2021

@author: lihongyi
"""
import numpy as np
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from data_schema import get_train_test_ml_set
import matplotlib.pyplot as plt
plt.ioff()


def train_ada_tree(independent_matrix, dependent_vector, depth, ccp_alpha=0, n_estimators=100):
    week = tree.DecisionTreeClassifier(criterion='entropy', max_depth=depth, ccp_alpha=ccp_alpha)
    clf = AdaBoostClassifier(week, n_estimators=n_estimators)
    clf = clf.fit(independent_matrix, dependent_vector)
    return clf


def error_rate(independent_matrix, dependent_vector, clf):
    pred = clf.predict(independent_matrix)
    miss = pred != dependent_vector
    err = sum(miss) / len(miss)
    return err


def cv_err_by_ada_tree_depth_alpha(train_set, depth, ccp_alpha=0, n_estimators=100, k=10):
    n = len(train_set.dependent_vector)
    
    errs_validate = []
    for i in range(k):
        idx_map = np.array([True] * n)
        if i == k-1:
            idx_map[(n // k)*i: ] = False
        else:
            idx_map[(n // k)*i: (n // k)*(i+1)] = False
        
        clf = train_ada_tree(train_set.independent_matrix[idx_map], train_set.dependent_vector[idx_map], depth, ccp_alpha=ccp_alpha, n_estimators=n_estimators)
        err_validate = error_rate(train_set.independent_matrix[~idx_map], train_set.dependent_vector[~idx_map], clf)
        errs_validate.append(err_validate)
    
    return np.mean(errs_validate)


def best_ada_tree_depth_by_cv(train_set, min_depth=1, max_depth=20, k=10, plot_name=None):
    depths = []
    cv_errs = []
    for d in range(min_depth, max_depth+1):
        depths.append(d)
        cv_errs.append(cv_err_by_ada_tree_depth_alpha(train_set, d, k=k))
        print(d)
    best_d = depths[np.argmin(np.round(cv_errs, 5))]
    
    if plot_name is not None:
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(depths, cv_errs)
        plt.axvline(x=best_d, color='red', linestyle='--')
        ax.set(xlabel='week tree max depth', ylabel='cross-validation mean error rate', 
               title=f'{k}-fold cross validation for Ada boosting on {plot_name}')
        fig.savefig(f'{plot_name}_ada_d_cv.png')
        plt.clf()
        plt.close()
    return best_d


def best_ada_ccp_alpha_by_cv(train_set, depth, min_log_alpha=-10, max_log_alpha=0, log_step=1.0, k=10, plot_name=None):
    ccp_alphas = []
    cv_errs = []
    for la in np.arange(min_log_alpha, max_log_alpha, log_step):
        a = np.exp(la)
        ccp_alphas.append(a)
        cv_errs.append(cv_err_by_ada_tree_depth_alpha(train_set, depth, ccp_alpha=a, k=k))
        print(la)
    best_a = ccp_alphas[np.argmin(np.round(cv_errs, 5))]
    
    if plot_name is not None:
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(ccp_alphas, cv_errs)
        plt.axvline(x=best_a, color='red', linestyle='--')
        ax.set(xlabel='week tree ccp alpha', ylabel=f'cross-validation mean error rate at depth {depth}', 
               title=f'{k}-fold cross validation for Ada boosting on {plot_name}')
        fig.savefig(f'{plot_name}_ada_a_cv.png')
        plt.clf()
        plt.close()
    return best_a


def run(data_set_name):
    train_set, test_set = get_train_test_ml_set(data_set_name)
    best_d = best_ada_tree_depth_by_cv(train_set, plot_name=data_set_name)
    best_a = best_ada_ccp_alpha_by_cv(train_set, best_d, plot_name=data_set_name)
    clf = train_ada_tree(train_set.independent_matrix, train_set.dependent_vector, best_d, ccp_alpha=best_a)
    train_err = error_rate(train_set.independent_matrix, train_set.dependent_vector, clf)
    with open(f'{data_set_name}_ada_train_err.txt', 'w') as f:
        f.write('%.5f' % train_err)
    test_err = error_rate(test_set.independent_matrix, test_set.dependent_vector, clf)
    with open(f'{data_set_name}_ada_test_err.txt', 'w') as f:
        f.write('%.5f' % test_err)


def main():
    run('abalone')
    run('bank-additional')


if __name__ == '__main__':
    main()
