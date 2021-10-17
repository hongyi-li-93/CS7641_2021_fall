#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 18:25:59 2021

@author: lihongyi
"""

import numpy as np
import mlrose_hiive
import matplotlib.pyplot as plt

import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose


rand_seed = 123


class MyDiscreteOpt(mlrose_hiive.DiscreteOpt):
    def eval_mate_probs_threshold(self, prob=0.95):
        """
        Calculate the probability of each member of the population reproducing.
        """
        pop_fitness = np.copy(self.pop_fitness)

        # Set -1*inf values to 0 to avoid dividing by sum of infinity.
        # This forces mate_probs for these pop members to 0.
        pop_fitness[pop_fitness == -1.0*np.inf] = 0

        if np.sum(pop_fitness) == 0:
            self.mate_probs = np.ones(len(pop_fitness)) \
                              / len(pop_fitness)
        else:
            med = np.quantile(pop_fitness, prob)
            where = np.where(pop_fitness >= med)
            self.mate_probs = pop_fitness * 0.0
            self.mate_probs[where] += 1.0 / len(where[0])


def get_edges(n_v, avg_e_per_v):
    assert n_v > 1
    assert avg_e_per_v < n_v
    n_tot = n_v * (n_v - 1) // 2
    n_pick = int(np.round(n_v * avg_e_per_v / 2))
    ids = np.random.RandomState(seed=rand_seed).permutation(n_tot)[:n_pick]
    edges = []
    for i in ids:
        k = int(np.floor(np.round(np.sqrt(2*(i+1)))))
        while k * (k + 1) < 2 * (1+i):
            k += 1
        c = int(np.round(k * (k - 1) / 2))
        s = i + 1 - c
        assert s < c
        edges.append((k, s))
    assert len(set(edges)) == n_pick
    return edges


def get_tree_edges(n_v, max_branch):
    assert n_v > 1
    assert max_branch < n_v
    ids = np.random.RandomState(seed=rand_seed).permutation(n_v).tolist()
    branches = np.random.RandomState(seed=rand_seed).randint(max_branch+1, size=(n_v,))
    edges = []
    frontier = []
    r = ids.pop()
    frontier.append(r)
    i = 0
    while len(frontier):
        cur = frontier.pop(0)
        k = branches[i]
        i += 1
        for _ in range(k):
            if len(ids) == 0:
                break
            child = ids.pop()
            edges.append((cur, child))
            frontier.append(child)
    return edges


problems = {
    '4peaks': MyDiscreteOpt(length = 100, fitness_fn=mlrose_hiive.FourPeaks(t_pct=0.2), maximize=True, max_val=2),
    'queens': MyDiscreteOpt(length = 100, fitness_fn=mlrose_hiive.Queens(), maximize=False, max_val=100),
    '3colors': MyDiscreteOpt(length = 100, fitness_fn=mlrose_hiive.MaxKColor(get_edges(100, 40)), maximize=False, max_val=3),
    '2colors': MyDiscreteOpt(length = 150, fitness_fn=mlrose_hiive.MaxKColor(get_tree_edges(150, 4)), maximize=False, max_val=2),
    'flip': MyDiscreteOpt(length = 200, fitness_fn=mlrose_hiive.FlipFlop(), maximize=True, max_val=2),
    }


def plot_fitness_curve(p_name, alg_name, best_state, best_fitness, curve, time):
    curve_raw = curve
    plt.plot(curve_raw) 
    plt.xlabel(f'nb. iteration for {alg_name}') 
    plt.ylabel(f'fitness for {p_name}') 
    plt.savefig(f'{p_name}_{alg_name}.png')
    plt.close()
    with open(f'{p_name}_{alg_name}.txt', "w") as text_file: 
        text_file.write(str(time))
        text_file.write('\n')
        text_file.write(str(best_fitness))
        text_file.write('\n')
        text_file.write(', '.join([str(i) for i in best_state]))
    