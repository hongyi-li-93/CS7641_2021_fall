#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 20:05:26 2021

@author: lihongyi
"""

import time
import numpy as np
from problems import mlrose_hiive, rand_seed, problems, plot_fitness_curve, plt
from data_schema import get_train_test_ml_set
from neural_net import train_network, error_rate, cv_err


def run():
    alg_name = 'simulated annealing geo decay'
    
    start = time.time()
    best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(
        problems['queens'], schedule=mlrose_hiive.GeomDecay(init_temp=5, decay=0.99997, min_temp=.0001),
        max_attempts=2000, max_iters=200000, init_state=None, 
        curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('queens', alg_name, best_state, best_fitness, curve[:, 0]*problems['queens'].maximize, t)

    start = time.time()
    best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(
        problems['4peaks'], schedule=mlrose_hiive.GeomDecay(init_temp=5, decay=0.99997, min_temp=.0001),
        max_attempts=2000, max_iters=200000, init_state=None, 
        curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('4peaks', alg_name, best_state, best_fitness, curve[:, 0]*problems['4peaks'].maximize, t)

    start = time.time()
    best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(
        problems['3colors'], schedule=mlrose_hiive.GeomDecay(init_temp=5, decay=0.99997, min_temp=.0001),
        max_attempts=1000, max_iters=100000, init_state=None, 
        curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('3colors', alg_name, best_state, best_fitness, curve[:, 0]*problems['3colors'].maximize, t)

    start = time.time()
    best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(
        problems['2colors'], schedule=mlrose_hiive.GeomDecay(init_temp=5, decay=0.99997, min_temp=.0001),
        max_attempts=2000, max_iters=200000, init_state=None, 
        curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('2colors', alg_name, best_state, best_fitness, curve[:, 0]*problems['2colors'].maximize, t)

    start = time.time()
    best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(
        problems['flip'], schedule=mlrose_hiive.GeomDecay(init_temp=5, decay=0.99997, min_temp=.0001),
        max_attempts=1000, max_iters=100000, init_state=None, 
        curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('flip', alg_name, best_state, best_fitness, curve[:, 0]*problems['flip'].maximize, t)
    
    alg_name = 'simulated annealing arith decay'
    
    start = time.time()
    best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(
        problems['queens'], schedule=mlrose_hiive.ArithDecay(init_temp=5.0, decay=5/200000, min_temp=0.0001),
        max_attempts=2000, max_iters=200000, init_state=None, 
        curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('queens', alg_name, best_state, best_fitness, curve[:, 0]*problems['queens'].maximize, t)
    
    start = time.time()
    best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(
        problems['4peaks'], schedule=mlrose_hiive.ArithDecay(init_temp=5.0, decay=5/200000, min_temp=0.0001),
        max_attempts=2000, max_iters=200000, init_state=None, 
        curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('4peaks', alg_name, best_state, best_fitness, curve[:, 0]*problems['4peaks'].maximize, t)
    
    start = time.time()
    best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(
        problems['3colors'], schedule=mlrose_hiive.ArithDecay(init_temp=5.0, decay=5/100000, min_temp=0.0001),
        max_attempts=1000, max_iters=100000, init_state=None, 
        curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('3colors', alg_name, best_state, best_fitness, curve[:, 0]*problems['3colors'].maximize, t)
    
    start = time.time()
    best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(
        problems['2colors'], schedule=mlrose_hiive.ArithDecay(init_temp=5.0, decay=5/100000, min_temp=0.0001),
        max_attempts=1000, max_iters=100000, init_state=None, 
        curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('2colors', alg_name, best_state, best_fitness, curve[:, 0]*problems['2colors'].maximize, t)
    
    start = time.time()
    best_state, best_fitness, curve = mlrose_hiive.simulated_annealing(
        problems['flip'], schedule=mlrose_hiive.ArithDecay(init_temp=5.0, decay=5/100000, min_temp=0.0001),
        max_attempts=1000, max_iters=100000, init_state=None, 
        curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('flip', alg_name, best_state, best_fitness, curve[:, 0]*problems['flip'].maximize, t)


def nn():
    algorithm = 'simulated_annealing'
    train_set, test_set = get_train_test_ml_set('abalone')
    max_iters=30000
    step_size=.02
    decays = ['geo 0.997', 'geo 0.9995', 'arith .001', 'arith .0005']
    schedules = {
        'geo 0.997': mlrose_hiive.GeomDecay(init_temp=1, decay=0.997, min_temp=.0001),
        'geo 0.9995': mlrose_hiive.GeomDecay(init_temp=1, decay=0.9995, min_temp=.0001),
        'arith .001': mlrose_hiive.ArithDecay(init_temp=1, decay=.001, min_temp=0.0001),
        'arith .0005': mlrose_hiive.ArithDecay(init_temp=1, decay=.0005, min_temp=0.0001),
        }
    
    errs = []
    costs = []
    curves = []
    for decay in decays:
        err, t_cost, curve = cv_err(
            train_set, algorithm, step_size, 
            max_iters=max_iters, schedule=schedules[decay])
        errs.append(err)
        costs.append(t_cost)
        curves.append(curve)
    
    for i, curve in enumerate(curves):
        plt.plot(curve, label=f'decay {decays[i]}')
    plt.xlabel("iterations ignoring attempt")
    plt.ylabel("fitness")
    plt.legend()
    plt.savefig(f'nn_fit_{algorithm}.png')
    plt.close()
    
    with open(f'nn_cv_{algorithm}.txt', "w") as text_file: 
        for err in errs:
            text_file.write(str(err))
            text_file.write('\n')
    
    with open(f'nn_time_{algorithm}.txt', "w") as text_file: 
        for t in costs:
            text_file.write(str(t))
            text_file.write('\n')
    
    decay = decays[np.argmin(errs)]
    scaler, clf, t, curve = train_network(
        train_set.independent_matrix, train_set.dependent_vector, 
        algorithm, step_size, 
        max_iters=max_iters, schedule=schedules[decay])
    
    train_err = error_rate(train_set.independent_matrix, train_set.dependent_vector, scaler, clf)
    test_err = error_rate(test_set.independent_matrix, test_set.dependent_vector, scaler, clf)
    with open(f'nn_{algorithm}.txt', "w") as text_file: 
        text_file.write(str(t))
        text_file.write('\n')
        text_file.write(str(train_err))
        text_file.write('\n')
        text_file.write(str(test_err))
