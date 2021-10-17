#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 20:29:07 2021

@author: lihongyi
"""

import numpy as np
import time
from problems import mlrose_hiive, rand_seed
from sklearn.preprocessing import StandardScaler


def train_network(independent_matrix, dependent_vector, 
                  algorithm, learning_rate, 
                  max_iters=1000, restarts=50, schedule=None, 
                  pop_size=500, mutation_prob=0.2):
    scaler = StandardScaler()
    scaler.fit(independent_matrix) 
    independent_matrix = scaler.transform(independent_matrix)
    
    clf = mlrose_hiive.NeuralNetwork(
        hidden_nodes=(50, ), activation='relu', 
        algorithm=algorithm, max_iters=max_iters, 
        bias=True, is_classifier=True, 
        learning_rate=learning_rate, early_stopping=True, clip_max=1000.0, 
        restarts=restarts, 
        schedule=schedule, 
        pop_size=pop_size, 
        mutation_prob=mutation_prob, 
        max_attempts=100, 
        random_state=rand_seed, curve=True)
    
    start = time.time()
    clf.fit(independent_matrix, dependent_vector)
    t = time.time() - start
    
    return scaler, clf, t, clf.fitness_curve[:,0]


def error_rate(independent_matrix, dependent_vector, scaler, clf):
    independent_matrix = scaler.transform(independent_matrix)
    pred = clf.predict(independent_matrix)[:, 0]
    miss = pred != dependent_vector
    err = sum(miss) / len(miss)
    return err


def cv_err(
        train_set, algorithm, learning_rate, 
        max_iters=1000, restarts=50, schedule=None, 
        pop_size=500, mutation_prob=0.2, k=3):
    n = len(train_set.dependent_vector)
    
    errs_validate = []
    time_costs = []
    curves = []
    for i in range(k):
        idx_map = np.array([True] * n)
        if i == k-1:
            idx_map[(n // k)*i: ] = False
        else:
            idx_map[(n // k)*i: (n // k)*(i+1)] = False
        
        scaler, clf, t, curve = train_network(
            train_set.independent_matrix[idx_map], train_set.dependent_vector[idx_map], 
            algorithm, learning_rate, 
            max_iters=max_iters, restarts=restarts, schedule=schedule, 
            pop_size=pop_size, mutation_prob=mutation_prob)
        err_validate = error_rate(train_set.independent_matrix[~idx_map], train_set.dependent_vector[~idx_map], scaler, clf)
        errs_validate.append(err_validate)
        time_costs.append(t)
        curves.append(curve)
        print(i)
    
    min_c_l = min([len(c) for c in curves])
    curve_m = np.array([c[:min_c_l] for c in curves]).mean(axis=0)
    return np.mean(errs_validate), np.mean(time_costs), curve_m
