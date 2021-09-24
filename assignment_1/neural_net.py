#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:29:07 2021

@author: lihongyi
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from data_schema import get_train_test_ml_set
import matplotlib.pyplot as plt
plt.ioff()


def train_network(independent_matrix, dependent_vector, 
                  n_hidden_nodes_first_layer, n_hidden_nodes_second_layer):
    scaler = StandardScaler()
    scaler.fit(independent_matrix) 
    independent_matrix = scaler.transform(independent_matrix)
    clf = MLPClassifier(solver='adam', 
                        activation='relu',
                        hidden_layer_sizes=(n_hidden_nodes_first_layer, 
                                            n_hidden_nodes_second_layer, ), 
                        random_state=1)
    clf.fit(independent_matrix, dependent_vector)
    return scaler, clf


def error_rate(independent_matrix, dependent_vector, scaler, clf):
    independent_matrix = scaler.transform(independent_matrix)
    pred = clf.predict(independent_matrix)
    miss = pred != dependent_vector
    err = sum(miss) / len(miss)
    return err


def cv_err_by_ns_hidden_nodes(train_set, 
                              n_hidden_nodes_first_layer, 
                              n_hidden_nodes_second_layer, 
                              k=10):
    n = len(train_set.dependent_vector)
    
    errs_validate = []
    for i in range(k):
        idx_map = np.array([True] * n)
        if i == k-1:
            idx_map[(n // k)*i: ] = False
        else:
            idx_map[(n // k)*i: (n // k)*(i+1)] = False
        
        scaler, clf = train_network(train_set.independent_matrix[idx_map], train_set.dependent_vector[idx_map], 
                                    n_hidden_nodes_first_layer, n_hidden_nodes_second_layer)
        err_validate = error_rate(train_set.independent_matrix[~idx_map], train_set.dependent_vector[~idx_map], scaler, clf)
        errs_validate.append(err_validate)
    
    return np.mean(errs_validate)


def best_network_ns_hidden_nodes_by_cv(train_set, ns_first_layer, ns_second_layer, 
                                       k=10, save_df_name=None):
    cv_errs = []
    min_err = 1.0
    best_ns = None
    for n_first_layer in ns_first_layer:
        errs = []
        for n_second_layer in ns_second_layer:
            err = cv_err_by_ns_hidden_nodes(train_set, n_first_layer, n_second_layer, k=k)
            errs.append(err)
            if err < min_err:
                min_err = err
                best_ns = n_first_layer, n_second_layer
            print(n_first_layer, n_second_layer)
        cv_errs.append(errs)
    
    if save_df_name is not None:
        df = pd.DataFrame(data=cv_errs, 
                          columns=ns_second_layer, index=ns_first_layer)
        df.columns.names = ['second hidden layer nb nodes']
        df.index.names = ['first hidden layer nb nodes']
        df.to_csv(f'{save_df_name}_nnet_cv.csv', index=True)
    
    return best_ns


def run(data_set_name):
    train_set, test_set = get_train_test_ml_set(data_set_name)
    n_first_layer, n_second_layer = best_network_ns_hidden_nodes_by_cv(
        train_set, 
        [10, 50, 90, 130], 
        [10, 40, 70, 100],
        save_df_name=data_set_name)
    scaler, clf = train_network(
        train_set.independent_matrix, 
        train_set.dependent_vector, 
        n_first_layer, 
        n_second_layer)
    test_err = error_rate(test_set.independent_matrix, test_set.dependent_vector, scaler, clf)
    with open(f'{data_set_name}_nnet_test_err.txt', 'w') as f:
        f.write('%.5f' % test_err)


def main():
    run('abalone')
    run('bank-additional')


if __name__ == '__main__':
    main()
