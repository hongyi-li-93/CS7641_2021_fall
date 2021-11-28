#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:53:43 2021

@author: lihongyi
"""

import gym
import numpy as np

n_space = 16
n_act = 4
n_search = 10000

env = gym.make('FrozenLake-v0')

trans = {}

for s in range(n_space):
    if s not in trans:
        trans[s] = {}
    for a in range(n_act):
        if a not in trans[s]:
            trans[s][a] = {}
        for _ in range(n_search):
            env.reset()
            env.env.s = s
            new_s, r, d, p = env.step(a)
            key = (new_s, r, d)
            if key not in trans[s][a]:
                trans[s][a][key] = 1
            else:
                trans[s][a][key] += 1
                

for s in range(n_space):
    for a in range(n_act):
        tps = trans[s][a]
        sum_p = 0
        for key, p in tps.items():
            tps[key] = np.round(3 * p / n_search) / 3
            sum_p += tps[key]
        np.testing.assert_almost_equal(sum_p, 1)
        

gamma = .99999

epsl = 1e-7
diff = 1
us = np.zeros(shape=(n_space,))
while diff > epsl:
    u_old = us.copy() + 0
    for s in range(n_space):
        candi = []
        for a, tps in trans[s].items():
            v = 0
            for (new_s, r, d), p in tps.items():
                v += (r * p)
                if not d:
                    v += (u_old[new_s] * p) * gamma
            candi.append(v)
        us[s] = np.max(candi)
    diff = np.abs(us - u_old).sum()

print(us[0])
