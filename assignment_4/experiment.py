#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 19:06:38 2021

@author: lihongyi
"""
import gym
import value_iteration
import policy_iteration
import double_dqn
from helper import play_game_comparison


gamma = 0.99


def adj_reward_fl(r, done, i):
    if done:
        if r > 0:
            return 1000
        if i < 100:
            return -10
    return -.1


def adj_reward_cp(r, done, i):
    if done and i < 200:
        return -200
    return r


env = gym.make('FrozenLake-v0')
max_iter = 2000
for d in [0, 0.9]:
    name = f'flake_value_iteration_decay{d}'
    value = value_iteration.run_value_iteration(env, gamma, discrete=True, decay_power=d, max_iter=max_iter, name=name)
    play_game_comparison(env, value, False, name=name)
    
    name = f'flake_policy_iteration_decay{d}'
    policy = policy_iteration.run_value_iteration(env, gamma, discrete=True, decay_power=d, max_iter=max_iter, name=name)
    play_game_comparison(env, policy, False, name=name)

name = 'flake_ddqn'
model, r_sums, m_wts = double_dqn.run_double_dqn(env, False, gamma, adj_reward_fl, learning_rate=0.002, replay_buffer_size=10000, 
                                                 avg_stop=0.75, decay=0.6, max_episodes=10000, minimum_epsilon=0.001, name=name)
model.set_weights(m_wts[-50])
play_game_comparison(env, model, False, name=name, ddqn=True)


env = gym.make('CartPole-v0')
max_iter = 1000
tol=0.01
fixed_hist_n_episode = 10000
name = 'cpole_value_iteration_fixed_hist'
value = value_iteration.run_value_iteration(env, gamma, discrete=False, decay_power=0, max_iter=max_iter, tol=tol, fixed_hist_n_episode=fixed_hist_n_episode, prt_freq=1, name=name)
value.set_test()
play_game_comparison(env, value, True, name=name)

name = 'cpole_policy_iteration_fixed_hist'
policy = policy_iteration.run_value_iteration(env, gamma, discrete=False, decay_power=0, max_iter=max_iter, tol=tol, fixed_hist_n_episode=fixed_hist_n_episode, name=name)
policy.set_test()
play_game_comparison(env, policy, True, name=name)

name = 'cpole_ddqn'
model, r_sums, m_wts = double_dqn.run_double_dqn(env, True, gamma, adj_reward_cp, epsilon_decay_factor_per_episode=0.993, name=name)
model.set_weights(m_wts[-50])
play_game_comparison(env, model, True, name=name, ddqn=True)
