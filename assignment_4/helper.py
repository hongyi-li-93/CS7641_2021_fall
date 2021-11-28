#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 08:37:14 2021

@author: lihongyi
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from sklearn.neighbors import KNeighborsRegressor
#from keras.regularizers import l2
#from keras.losses import Huber, Reduction

seed = 123
n_epochs = 2
max_steps = 1000


def iteration_step(old_s, a, r, new_s, stop, status_func, hist, cont=False):
    old_s = status_func(old_s)
    new_s = status_func(new_s)
    
    if cont:
        hist['old_s'].append(old_s)
        hist['a'].append(a)
        hist['new_s'].append(new_s)
        hist['r'].append(r)
        hist['stop'].append(stop)
    else:
        key = (old_s, a)
        if key not in hist:
            hist[key] = []
        hist[key].append([new_s, r, stop])


def iteration_episode(env, status_func, hist, cont=False, early_stop=True):
    old_s = env.reset()
    stop = False
    for _ in range(500):
        a = np.random.randint(env.action_space.n)
        new_s, r, new_stop, _ = env.step(a)
        iteration_step(old_s, a, r, new_s, stop, status_func, hist, cont=cont)
        old_s = new_s
        stop = new_stop
        if stop and early_stop:
            iteration_step(old_s, 0, 0, old_s, stop, status_func, hist, cont=cont)
            return


def discrete_value_grid(n_space, n_action):
    grid = np.zeros(shape=(n_space, n_action))
    return grid


def get_discrete_status(s):
    return s


def get_mat_status(s):
    return np.array([s])


def get_vector_status(s, cont, fraction_finished, input_dim):
    if cont:
        return np.append(s, fraction_finished)
    else:
        sv = np.zeros(shape=(input_dim, ))
        sv[s] = 1
        sv[-1] = fraction_finished
        return sv


def init_hist(cont):
    if cont:
        hist = {
            'old_s': [],
            'a': [],
            'new_s': [],
            'r': [],
            'stop': [],
            }
    else:
        hist = {}
    return hist


def update_grid_inplace(grid, hist, nw, u_func, gamma, cont=False, tol=0.00001, policy_stop_tol=None, rec=None, prt=False):
    pol_diff = None
    if cont:
        new_s_m = np.concatenate(hist['new_s'], axis=0)
        old_s_m = np.concatenate(hist['old_s'], axis=0)
        rs = np.array(hist['r'])
        n = len(rs)
        
        u_new = u_func(grid, new_s_m, rs, gamma, cont)
        
        grid_ys = grid.predict(old_s_m)
        grid_ys_new = grid_ys.copy() + 0
        grid_ys_new[np.arange(n), hist['a']] = u_new
        grid_ys_new[hist['stop'], :] = 0
        diff_ys = (grid_ys_new - grid_ys) * nw
        
        diff = np.abs(diff_ys).mean()
        if diff < tol:
            return True
        
        if policy_stop_tol is not None:
            pol_diff = np.sum(grid_ys_new.argmax(axis=1) != grid_ys.argmax(axis=1))/n
            if pol_diff < policy_stop_tol:
                return True
        
        grid_ys += diff_ys
        grid.fit(old_s_m, grid_ys, epochs=n_epochs)
        
    else:
        old_grid = grid.copy() + 0
        for (s, a), obs in hist.items():
            m = len(obs)
            
            u_new = u_func(grid, [o[0] for o in obs], np.array([o[1] for o in obs]), gamma, cont)
            
            freq_stop = np.sum([o[2] for o in obs])/m
            u_new *= (1-freq_stop)
            grid[s, a] += nw * (u_new - grid[s, a])
        
        diff = np.abs(grid - old_grid).mean()
        if diff < tol:
            return True
        
        if policy_stop_tol is not None: 
            pol_diff = np.sum(grid.argmax(axis=1) != old_grid.argmax(axis=1))/grid.shape[0]
            if pol_diff < policy_stop_tol:
                print(pol_diff, policy_stop_tol)
                return True
    
    if rec is not None:
        rec.append((diff, pol_diff))
    if prt:
        print(diff)
        if policy_stop_tol is not None: 
            print(pol_diff)
    return False


def get_cont_grid(input_dim, n_action, learning_rate=0.001):
    optimizer = Adam(learning_rate=learning_rate)
    #optimizer = RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01)
    #loss = Huber(delta=1.0, reduction=Reduction.SUM)
    loss = 'mse'
    grid = Sequential([
      Dense(16, input_dim=input_dim, activation="relu", ),
      Dense(16, activation="relu", ),
      Dense(n_action, activation='linear', )
    ])
    grid.compile(optimizer=optimizer, loss=loss, metrics=[])
    
    return grid


class KNN_regressor:
    def __init__(self, input_dim, n_action, fixed_samples=False):
        self.input_dim = input_dim
        self.n_action = n_action # I was using input_dim
        self.models = None
        self.weights = None
        self.fixed_samples = fixed_samples
        self.fixed_model = None
        self.xso_ids = None
        self.xsp_ids = None
        self.y_old = None
        self.y_pred = None
        self.test=False
    
    def set_test(self):
        self.test = True
    
    def fit(self, xs, ys, *args, **kwargs):
        assert xs.shape[1] == self.input_dim
        assert ys.shape[1] == self.n_action
        assert xs.shape[0] == ys.shape[0]
        self.weights = xs, ys
        
        if self.fixed_samples:
            self.y_pred = None
            if self.fixed_model is None:
                self.fixed_model = KNeighborsRegressor(n_neighbors=2**self.input_dim)
                self.fixed_model.fit(xs, ys)
                self.xso_ids = self.fixed_model.kneighbors(xs)[1]
            else:
                self.fixed_model._y = ys
            self.y_old = np.mean(self.fixed_model._y[self.xso_ids], axis=1)
            return
        
        self.models = []
        for a in range(self.n_action):
            neigh = KNeighborsRegressor(n_neighbors=2**self.input_dim)
            neigh.fit(xs, ys[:, a])
            self.models.append(neigh)
    
    def predict(self, xs):
        assert xs.shape[1] == self.input_dim
        ys = np.zeros(shape=(xs.shape[0], self.n_action))
        
        if self.fixed_samples:
            if self.test:
                return self.fixed_model.predict(xs)
            
            if self.fixed_model is None:
                return ys
            if np.abs(xs - self.weights[0]).sum() < 1e-7:
                return self.y_old 
            if self.y_pred is not None:
                return self.y_pred
            if self.xsp_ids is None:
                self.xsp_ids = self.fixed_model.kneighbors(xs)[1]
            self.y_pred = np.mean(self.fixed_model._y[self.xsp_ids], axis=1)
            return self.y_pred
        
        if self.models is None:
            return ys
        for a in range(self.n_action):
            neigh = self.models[a]
            ys[:, a] = neigh.predict(xs)
        return ys
    
    def get_weights(self):
        return self.weights
          

def get_knn_grid(input_dim, n_action, fixed_samples=False):
    grid = KNN_regressor(input_dim, n_action, fixed_samples=fixed_samples)
    return grid


def play_game_episode(env, grid, rand_play, cont, render=False, ddqn_input_dim=None):
    
    
    s = env.reset()
    u = 0
    stop = False
    fraction_finished = 0.0
    for t in range(max_steps):
        fraction_finished = (t+1) / max_steps
        if rand_play:
            a = np.random.randint(env.action_space.n)
        else:
            if ddqn_input_dim is not None: 
                s = get_vector_status(s, cont, fraction_finished, ddqn_input_dim)
                a = grid.predict(np.array([s]))[0].argmax()
            elif cont:
                a = grid.predict(np.array([s]))[0].argmax()
            else:
                a = grid[s, :].argmax()
        if render:
            env.render()
        new_s, r, stop, _ = env.step(a)
        u += r
        s = new_s      
        if stop:
            break
    return u


def play_game_comparison(env, grid, cont, n_episode=2000, render=False, r_func=get_vector_status, ddqn=False, name=None):
    if cont:
        input_dim = len(env.reset()) + 1
    else:
        input_dim = env.observation_space.n + 1
    if not ddqn:
        input_dim = None
    
    u_policy = 0
    u_ref = 0
    for _ in range(n_episode):
        u_policy += play_game_episode(env, grid, False, cont, render, ddqn_input_dim=input_dim)
        u_ref += play_game_episode(env, grid, True, cont)
        print((_, u_policy))
    
    u_policy /= n_episode
    u_ref /= n_episode
    
    if name is not None:
        with open(f'{name}_comparison.txt', 'w') as f:
            f.write('%.5f' % u_policy)
            f.write('\n')
            f.write('%.5f' % u_ref)
    return u_policy, u_ref


def iden_reward(r, *args):
    return r
