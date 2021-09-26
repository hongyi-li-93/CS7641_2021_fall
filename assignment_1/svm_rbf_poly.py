#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 12:11:59 2021

@author: lihongyi
"""
import numpy as np
import scipy as sp
from sklearn import svm
from data_schema import get_train_test_ml_set
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.ioff()


def train_svm(independent_matrix, dependent_vector, kernel, degree=3, gamma='scale'):
    assert kernel in {'rbf', 'poly'}, 'I donnot want other kernels.'
    
    scaler = StandardScaler()
    scaler.fit(independent_matrix) 
    independent_matrix = scaler.transform(independent_matrix)
    
    clf = svm.SVC(kernel=kernel, degree=degree, gamma=gamma)
    clf = clf.fit(independent_matrix, dependent_vector)
    return scaler, clf


def error_rate(independent_matrix, dependent_vector, scaler, clf):
    independent_matrix = scaler.transform(independent_matrix)
    pred = clf.predict(independent_matrix)
    miss = pred != dependent_vector
    err = sum(miss) / len(miss)
    return err


def cv_err_by_svm_kernel(train_set, kernel, degree=3, gamma='scale', k=10):
    n = len(train_set.dependent_vector)
    
    errs_validate = []
    for i in range(k):
        idx_map = np.array([True] * n)
        if i == k-1:
            idx_map[(n // k)*i: ] = False
        else:
            idx_map[(n // k)*i: (n // k)*(i+1)] = False
        
        scaler, clf = train_svm(train_set.independent_matrix[idx_map], train_set.dependent_vector[idx_map], kernel, degree=degree, gamma=gamma)
        err_validate = error_rate(train_set.independent_matrix[~idx_map], train_set.dependent_vector[~idx_map], scaler, clf)
        errs_validate.append(err_validate)
        print(f'fold{i}')
    
    return np.mean(errs_validate)


def best_rbf_gamma_by_cv(train_set, min_log_gamma=-6, max_log_gamma=3, log_step=.5, k=10, plot_name=None):
    gammas = []
    cv_errs = []
    for lg in np.arange(min_log_gamma, max_log_gamma, log_step):
        g = np.exp(lg)
        gammas.append(g)
        cv_errs.append(cv_err_by_svm_kernel(train_set, kernel='rbf', gamma=g, k=k))
        print(lg)
    best_g = gammas[np.argmin(np.round(cv_errs, 5))]
    
    if plot_name is not None:
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(np.log(gammas), cv_errs)
        plt.axvline(x=np.log(best_g), color='red', linestyle='--')
        ax.set(xlabel='log gamma for rbf', ylabel='cross-validation mean error rate', 
               title=f'{k}-fold cross validation for svm on {plot_name}')
        fig.savefig(f'{plot_name}_svm_rbf_cv.png')
        plt.clf()
        plt.close()
    return best_g


def best_poly_degree_by_cv(train_set, min_degree=0, max_degree=10, k=10, plot_name=None):
    degrees = []
    cv_errs = []
    for d in range(min_degree, max_degree+1):
        degrees.append(d)
        cv_errs.append(cv_err_by_svm_kernel(train_set, kernel='poly', degree=d, k=k))
        print(d)
    best_d = degrees[np.argmin(np.round(cv_errs, 5))]
    
    if plot_name is not None:
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(degrees, cv_errs)
        plt.axvline(x=best_d, color='red', linestyle='--')
        ax.set(xlabel='degree for poly', ylabel='cross-validation mean error rate', 
               title=f'{k}-fold cross validation for svm on {plot_name}')
        fig.savefig(f'{plot_name}_svm_poly_cv.png')
        plt.clf()
        plt.close()
    return best_d


def run(data_set_name):
    train_set, test_set = get_train_test_ml_set(data_set_name)
    
    best_g = best_rbf_gamma_by_cv(train_set, plot_name=data_set_name)
    scaler, clf = train_svm(train_set.independent_matrix, train_set.dependent_vector, kernel='rbf', gamma=best_g)
    train_err = error_rate(train_set.independent_matrix, train_set.dependent_vector, scaler, clf)
    with open(f'{data_set_name}_svm_rbf_train_err.txt', 'w') as f:
        f.write('%.5f' % train_err)
    test_err = error_rate(test_set.independent_matrix, test_set.dependent_vector, scaler, clf)
    with open(f'{data_set_name}_svm_rbf_test_err.txt', 'w') as f:
        f.write('%.5f' % test_err)
    
    best_d = best_poly_degree_by_cv(train_set, plot_name=data_set_name)
    scaler, clf = train_svm(train_set.independent_matrix, train_set.dependent_vector, kernel='poly', degree=best_d)
    train_err = error_rate(train_set.independent_matrix, train_set.dependent_vector, scaler, clf)
    with open(f'{data_set_name}_svm_poly_train_err.txt', 'w') as f:
        f.write('%.5f' % train_err)  
    test_err = error_rate(test_set.independent_matrix, test_set.dependent_vector, scaler, clf)
    with open(f'{data_set_name}_svm_poly_test_err.txt', 'w') as f:
        f.write('%.5f' % test_err)  


def main():
    run('abalone')
    run('bank-additional')


if __name__ == '__main__':
    main()
