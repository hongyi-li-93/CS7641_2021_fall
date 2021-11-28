#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:10:18 2021

@author: lihongyi
"""

import numpy as np
import tensorflow as tf
import pickle
import time
from helper import seed, get_cont_grid, get_vector_status, max_steps


training_start = 250
target_network_replace_frequency_steps = 200
starting_epsilon = 1.0
train_every_x_steps = 1


def update_target_model(behav_model, tgt_model):
    tgt_model.set_weights(behav_model.get_weights())


def select_action_epsilon_greedy(q_values, epsilon):
    random_value = np.random.random()
    if random_value < epsilon: 
        return np.random.randint(len(q_values))
    else:
        return np.argmax(q_values)


class ReplayBuffer():
  current_index = 0

  def __init__(self, size):
    self.size = size
    self.transitions = []
    self.forever_memory = int(size / 5)
    self.current_index = self.forever_memory

  def add(self, transition):
    if len(self.transitions) < self.size: 
      self.transitions.append(transition)
    else:
      self.transitions[self.current_index] = transition
      self.__increment_current_index()

  def length(self):
    return len(self.transitions)

  def get_batch(self, batch_size):
    return np.random.choice(self.transitions, batch_size, replace=True)

  def __increment_current_index(self):
    self.current_index += 1
    if self.current_index >= self.size: 
      self.current_index = self.forever_memory


def replay_d(minibatch, gamma, behav_model, tgt_model, nw=1):
    s_l = np.array([m['s'] for m in minibatch])
    a_l = np.array([m['a'] for m in minibatch])
    r_l = np.array([m['r'] for m in minibatch])
    s_new_l = np.array([m['s_new'] for m in minibatch])
    done_l = np.array([m['done'] for m in minibatch])
    
    qvals_new_l = tgt_model.predict(s_new_l)
    
    qvals_new_la = behav_model.predict(s_new_l)
    target_f = behav_model.predict(s_l).astype(np.float64) + 0.0
    for i, (s, a, r, qvals_a, qvals_new, done) in enumerate(zip(s_l, a_l, r_l, qvals_new_la, qvals_new_l, done_l)):
        if done:
            target = r
        else:
            aprime = np.argmax(qvals_a)
            target = target_f[i][a] + (r + gamma * qvals_new[aprime] - target_f[i][a]) * nw
        target_f[i][a] = target
    
    behav_model.fit(s_l, target_f, epochs=1, batch_size=len(target_f), verbose=0)


def run_double_dqn(env, cont, gamma, reward_func, learning_rate=0.001, training_batch_size=128, epsilon_decay_factor_per_episode=0.995, 
                   replay_buffer_size=2000, decay=0, max_episodes=2000, minimum_epsilon=0.05, avg_stop=195, render=False, name=None):
    start = time.time()
    env.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    replay_buffer = ReplayBuffer(replay_buffer_size)
    epsilon = starting_epsilon
    n_action = env.action_space.n
    if cont:
        input_dim = len(env.reset()) + 1
    else:
        input_dim = env.observation_space.n + 1
    
    model = get_cont_grid(input_dim, n_action, learning_rate=learning_rate)
    target_model = get_cont_grid(input_dim, n_action, learning_rate=learning_rate)
    
    m_wts = []
    r_sums = []
    ci = 0
    average = 0
    for n in range(max_episodes):
        print(f"Starting episode {n} with epsilon {epsilon}")
    
        r_sum = 0
        s = env.reset()
        fraction_finished = 0.0
        s = get_vector_status(s, cont, fraction_finished, input_dim)
        done = False
        i = 0
        while not done:
            if n > 300 and render:
                env.render()
        
            i += 1
            ci += 1
            q_values = model.predict(s.reshape(1, input_dim))[0]
            a = select_action_epsilon_greedy(q_values, epsilon)
            s_new, r_raw, done, _ = env.step(a)
            r = reward_func(r_raw, done, i)
            
            fraction_finished = i / max_steps
            s_new = get_vector_status(s_new, cont, fraction_finished, input_dim)
            
            r_sum += r_raw
                
            state_transition = {
                's': s,
                'a': a,
                'r': r,
                's_new': s_new,
                'done': done,
                }
            replay_buffer.add(state_transition)
        
            s = s_new

            if ci % target_network_replace_frequency_steps == 0:
                print("Updating target model")
                update_target_model(model, target_model)

            if replay_buffer.length() >= training_start and ci % train_every_x_steps == 0:
                batch = replay_buffer.get_batch(batch_size=training_batch_size)
                nw = n ** (-decay)
                replay_d(batch, gamma, model, target_model, nw=nw) #new

            if i > max_steps:
                print(f"Episode reached the maximum number of steps. {max_steps}")
                break

        r_sums.append(r_sum)
        average = np.mean(r_sums[-100:])
        m_wts.append(model.get_weights())

        print(
          f"episode {n} finished in {i} steps with reward {r_sum}. "
          f"Average reward over last 100: {average}")
        if average >= avg_stop:
            break
            
        if replay_buffer.length() >= training_start:
            epsilon *= epsilon_decay_factor_per_episode
            epsilon = max(minimum_epsilon, epsilon)
    
    end = time.time()
    if name is not None:
        with open(f'{name}_cost.txt', 'w') as f:
            f.write('%.5f' % n)
            f.write('\n')
            f.write('%.5f' % (end-start))
        with open(f'{name}_rs_wts.pkl', 'wb') as p:
            pickle.dump((r_sums, m_wts), p)
            
    return model, r_sums, m_wts
    