#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 18:31:49 2021

@author: lihongyi
"""

import numpy as np
import pickle
import time
import tensorflow as tf
from helper import seed, discrete_value_grid, get_discrete_status, update_grid_inplace, get_cont_grid, init_hist, get_mat_status, iteration_episode, get_knn_grid

policy_matching_tol = 0.05
u_max = 200


def value_u_func(grid, new_s, rs, gamma, cont):
    if cont:
        u_new = np.minimum(grid.predict(new_s).max(axis=1) * gamma, u_max) + rs
        return u_new
    else:
        u_new = grid[new_s].max(axis=1) * gamma + rs
        return u_new.mean()


def update_grid_by_hist(env, grid, status_func, gamma, nw, n_episode=5000, hist=None, cont=False, tol=0.00001, policy_stop_tol=None, rec=None, prt=False):
    if hist is None:
        hist = init_hist(cont)
        for i  in range(n_episode):
            iteration_episode(env, status_func, hist, cont=cont)
    
    stop = update_grid_inplace(grid, hist, nw, value_u_func, gamma, cont=cont, tol=tol, policy_stop_tol=policy_stop_tol, rec=rec, prt=prt)
    return stop


def run_value_iteration(env, gamma, discrete=True, decay_power=0.0, max_iter=1000, tol=0.00001, fixed_hist_n_episode=None, prt_freq=50, name=None):
    assert 0 <= decay_power <= 1
    start = time.time()
    policy_stop_tol = None
    np.random.seed(seed)
    env.seed(seed)
    tf.random.set_seed(seed)
    n_action = env.action_space.n
    
    if discrete:
        status_func = get_discrete_status 
        n_space = env.observation_space.n 
        grid = discrete_value_grid(n_space, n_action)
        cont = False
        
    else:
        status_func = get_mat_status
        grid = get_knn_grid(len(env.reset()), n_action, fixed_samples=(fixed_hist_n_episode is not None))
        cont = True
    
    if decay_power <= 0.5:
        policy_stop_tol = policy_matching_tol 
    
    fixed_hist = None
    if fixed_hist_n_episode is not None:
        fixed_hist = init_hist(cont)
        for _ in range(fixed_hist_n_episode):
            iteration_episode(env, status_func, fixed_hist, cont=cont)
    
    rec = []
    for i in range(max_iter):
        nw = (i+1) ** (-decay_power)
        prt = False
        if i % prt_freq == 0:
            prt = True
        stop = update_grid_by_hist(env, grid, status_func, gamma, nw, 
                                   cont=cont, tol=tol, hist=fixed_hist,
                                   policy_stop_tol=policy_stop_tol, 
                                   prt=prt, rec=rec)
        if stop:
            break
    end = time.time()
    
    if name is not None:
        with open(f'{name}_cost.txt', 'w') as f:
            f.write('%.5f' % i)
            f.write('\n')
            f.write('%.5f' % (end-start))
        with open(f'{name}_conv_rec.pkl', 'wb') as p:
            pickle.dump(rec, p)
        with open(f'{name}_grid.pkl', 'wb') as p:
            if cont:
                pickle.dump(grid.get_weights(), p)
            else:
                pickle.dump(grid, p)
    return grid
