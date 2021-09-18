#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:29:07 2021

@author: lihongyi
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from data_schema import get_train_test_ml_set
import matplotlib.pyplot as plt
plt.ioff()


def train_network(independent_matrix, dependent_vector, n_hidden_nodes):
    scaler = StandardScaler()
    scaler.fit(independent_matrix) 
    independent_matrix = scaler.transform(independent_matrix)
    clf = MLPClassifier(solver='lbfgs', 
                        activation='logistic',
                        hidden_layer_sizes=(n_hidden_nodes, ), 
                        random_state=1)
    clf.fit(independent_matrix, dependent_vector)
    return scaler, clf


def error_rate(independent_matrix, dependent_vector, scaler, clf):
    independent_matrix = scaler.transform(independent_matrix)
    pred = clf.predict(independent_matrix)
    miss = pred != dependent_vector
    err = sum(miss) / len(miss)
    return err


def cv_err_by_n_hidden_nodes(train_set, n_hidden_nodes, k=10):
    n = len(train_set.dependent_vector)
    
    errs_validate = []
    for i in range(k):
        idx_map = np.array([True] * n)
        if i == k-1:
            idx_map[(n // k)*i: ] = False
        else:
            idx_map[(n // k)*i: (n // k)*(i+1)] = False
        
        scaler, clf = train_network(train_set.independent_matrix[idx_map], train_set.dependent_vector[idx_map], n_hidden_nodes)
        err_validate = error_rate(train_set.independent_matrix[~idx_map], train_set.dependent_vector[~idx_map], scaler, clf)
        errs_validate.append(err_validate)
    
    return np.mean(errs_validate)


def best_network_n_hidden_nodes_by_cv(train_set, min_n=1, max_n=50, k=10, step=2, plot_name=None):
    ns_nodes = []
    cv_errs = []
    for nd in range(min_n, max_n+step, step):
        ns_nodes.append(nd)
        cv_errs.append(cv_err_by_n_hidden_nodes(train_set, nd, k=k))
        #print(nd)
    best_nd = ns_nodes[np.argmin(np.round(cv_errs, 5))]
    
    if plot_name is not None:
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(ns_nodes, cv_errs)
        plt.axvline(x=best_nd, color='red', linestyle='--')
        ax.set(xlabel='nb of hidden nodes', ylabel='cross-validation mean error rate', 
               title=f'{k}-fold cross validation for neural network on {plot_name}')
        fig.savefig(f'{plot_name}_nnet_cv.png')
        plt.clf()
        plt.close()
    return best_nd


def run(data_set_name):
    train_set, test_set = get_train_test_ml_set(data_set_name)
    best_nd = best_network_n_hidden_nodes_by_cv(train_set, plot_name=data_set_name)
    scaler, clf = train_network(train_set.independent_matrix, train_set.dependent_vector, best_nd)
    test_err = error_rate(test_set.independent_matrix, test_set.dependent_vector, scaler, clf)
    with open(f'{data_set_name}_nnet_test_err.txt', 'w') as f:
        f.write('%d' % test_err)


def main():
    run('abalone')


if __name__ == '__main__':
    main()
