#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 18:35:13 2021

@author: lihongyi
"""

import numpy as np
import pickle
import time
import tensorflow as tf
from helper import seed, get_cont_grid, get_discrete_status, init_hist, iteration_episode, discrete_value_grid, update_grid_inplace, get_mat_status, get_knn_grid

policy_matching_tol = 0.05


def update_grid_by_hist(env, policy_grid, status_func, gamma, n_episode=5000, cont=False, tol=0.00001, decay_power=0.9, rec=None, max_iter=1000, fixed_hist=None):
    assert 0 <= decay_power <= 1
    policy_stop_tol=None
    if decay_power <= 0.5:
        policy_stop_tol = policy_matching_tol
    
    def policy_u_func(grid, new_s, rs, gamma, cont):
        n = len(rs)
        if cont:
            a_policy = policy_grid.predict(new_s).argmax(axis=1)
            u_new = grid.predict(new_s)[np.arange(n), a_policy] * gamma + rs
            return u_new
        else:
            a_policy = policy_grid[new_s, :].argmax(axis=1)
            u_new = grid[new_s, a_policy] * gamma + rs
            return u_new.mean()
    
    if cont:
        value_grid = get_knn_grid(len(env.reset()), env.action_space.n, fixed_samples=(fixed_hist is not None))
    else:
        value_grid = discrete_value_grid(policy_grid.shape[0], env.action_space.n)
        
    for i in range(max_iter):
        if fixed_hist is None:
            hist = init_hist(cont)
            for _  in range(n_episode):
                iteration_episode(env, status_func, hist, cont=cont)
        else:
            hist = fixed_hist
        
        nw = (i+1) ** (-decay_power)
        prt = False
        if i % 10 == 0:
            prt = True
        
        stop = update_grid_inplace(value_grid, hist, nw, policy_u_func, gamma, cont=cont, tol=tol, policy_stop_tol=policy_stop_tol, prt=prt)
        if stop:
            break
    if rec is not None:
        rec.append([i])
    return value_grid
    

def check_policy_match(grid_old, grid_new, env, status_func, cont, n_episode=1000, rec=None, tol=policy_matching_tol, hist=None):
    if not cont:
        pol_diff = np.sum(grid_old.argmax(axis=1) != grid_new.argmax(axis=1))/grid_old.shape[0]
        print(pol_diff)
        if rec is not None:
            rec[-1].append(pol_diff)
        return pol_diff < tol
    
    if hist is None:
        hist = init_hist(cont)
        for _  in range(n_episode):
            iteration_episode(env, status_func, hist, cont=cont)
    
    old_s_m = np.concatenate(hist['old_s'], axis=0)
    grid_old_ys = grid_old.predict(old_s_m)
    grid_new_ys = grid_new.predict(old_s_m)
    pol_diff = np.sum(grid_old_ys.argmax(axis=1) != grid_new_ys.argmax(axis=1))/grid_old_ys.shape[0]
    print(pol_diff)
    if rec is not None:
        rec[-1].append(pol_diff)
    return pol_diff < tol


def run_value_iteration(env, gamma, discrete=True, n_episode=1000, decay_power=0.9, tol=0.00001, max_iter=20, fixed_hist_n_episode=None, name=None):
    assert 0 <= decay_power <= 1
    start = time.time()
    np.random.seed(seed)
    env.seed(seed)
    tf.random.set_seed(seed)
    n_action = env.action_space.n
    
    if discrete:
        status_func = get_discrete_status 
        n_space = env.observation_space.n 
        policy_grid = discrete_value_grid(n_space, n_action)
        cont = False
        
    else:
        status_func = get_mat_status
        policy_grid = get_knn_grid(len(env.reset()), n_action, fixed_samples=(fixed_hist_n_episode is not None))
        cont = True
    
    fixed_hist = None
    if fixed_hist_n_episode is not None:
        fixed_hist = init_hist(cont)
        for _ in range(fixed_hist_n_episode):
            iteration_episode(env, status_func, fixed_hist, cont=cont)
    
    rec = []
    for i in range(max_iter):
        new_grid = update_grid_by_hist(env, policy_grid, status_func, gamma, cont=cont, tol=tol, rec=rec, decay_power=decay_power, fixed_hist=fixed_hist)
        if check_policy_match(policy_grid, new_grid, env, status_func, cont, rec=rec, hist=fixed_hist):
            break
        policy_grid = new_grid
        print(i)
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
                pickle.dump(policy_grid.get_weights(), p)
            else:
                pickle.dump(policy_grid, p)
    return policy_grid
