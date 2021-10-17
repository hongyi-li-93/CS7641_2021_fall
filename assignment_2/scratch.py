#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 08:02:30 2021

@author: lihongyi
"""



best_state, best_fitness, curve = mlrose.mimic(
    problems['3colors'], pop_size=9000, keep_pct=0.07, max_attempts=10, 
    max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)

plt.plot(curve)
best_fitness

best_state, best_fitness, curve = mlrose.mimic(
    problems['3colors'], pop_size=9000, keep_pct=0.1, max_attempts=10, 
    max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)

plt.plot(curve)
best_fitness

best_state, best_fitness, curve = mlrose.mimic(
    problems['3colors'], pop_size=5000, keep_pct=0.1, max_attempts=10, 
    max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)

plt.plot(curve)
best_fitness

problems = {
    '4peaks': MyDiscreteOpt(length = 100, fitness_fn=mlrose.FourPeaks(t_pct=0.2), maximize=True, max_val=2),
    'queens': mlrose.DiscreteOpt(length = 100, fitness_fn=mlrose.Queens(), maximize=False, max_val=100),
    '3colors': mlrose.DiscreteOpt(length = 100, fitness_fn=mlrose.MaxKColor(get_edges(100, avg_e_per_v=40)), maximize=False, max_val=3),
    }

best_state, best_fitness, curve = mlrose.mimic(
    problems['3colors'], pop_size=500, keep_pct=0.1, max_attempts=10, 
    max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)

plt.plot(curve)
best_fitness

best_state, best_fitness, curve = mlrose.simulated_annealing(
    problems['3colors'], schedule=mlrose.GeomDecay(init_temp=1, decay=0.9999, min_temp=.0001),
    max_attempts=500, max_iters=100000, init_state=None, 
    curve=True, random_state=rand_seed)

plt.plot(curve)
best_fitness

best_state, best_fitness, curve = mlrose.mimic(
    problems['3colors'], pop_size=1000, keep_pct=0.1, max_attempts=10, 
    max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)

plt.plot(curve)
best_fitness

best_state, best_fitness, curve = mlrose.mimic(
    problems['3colors'], pop_size=1000, keep_pct=0.15, max_attempts=10, 
    max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)

plt.plot(curve)
best_fitness

best_state, best_fitness, curve = mlrose.mimic(
    problems['3colors'], pop_size=1000, keep_pct=0.05, max_attempts=10, 
    max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)

plt.plot(curve)

best_state, best_fitness, curve = mlrose.mimic(
    problems['3colors'], pop_size=2000, keep_pct=0.05, max_attempts=10, 
    max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)

plt.plot(curve)
best_fitness,

best_state, best_fitness, curve = mlrose.mimic(
    problems['3colors'], pop_size=2000, keep_pct=0.1, max_attempts=10, 
    max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)

plt.plot(curve)
best_fitness

best_state, best_fitness, curve = mlrose.mimic(
    problems['3colors'], pop_size=2000, keep_pct=0.15, max_attempts=10, 
    max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)

plt.plot(curve)
best_fitness

sizes = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
pcts = [.03, .05, .07, .1, .15, .2]
tsts = []
for size in sizes:
    for pct in pcts:
        best_state, best_fitness, curve = mlrose.mimic( 
            problems['3colors'], pop_size=size, keep_pct=pct, max_attempts=10, 
            max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)
        print(size, pct, best_fitness)
        tsts.append((size, pct, best_fitness))