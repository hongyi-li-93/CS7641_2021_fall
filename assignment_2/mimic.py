#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 20:55:31 2021

@author: lihongyi
"""

import numpy as np
import time
from problems import mlrose_hiive, rand_seed, problems, plot_fitness_curve, mlrose


alg_name = 'mimic'


def mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10,
          max_iters=np.inf, curve=False, random_state=None, fast_mimic=False):
    """Use MIMIC to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()` or :code:`TSPOpt()`.
    pop_size: int, default: 200
        Size of population to be used in algorithm.
    keep_pct: float, default: 0.2
        Proportion of samples to keep at each iteration of the algorithm,
        expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    curve: bool, default: False
        Boolean to keep fitness values for a curve.
        If :code:`False`, then no curve is stored.
        If :code:`True`, then a history of fitness values is provided as a
        third return value.
    random_state: int, default: None
        If random_state is a positive integer, random_state is the seed used
        by np.random.seed(); otherwise, the random seed is not set.
    fast_mimic: bool, default: False
        Activate fast mimic mode to compute the mutual information in
        vectorized form. Faster speed but requires more memory.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array containing the fitness at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.

    References
    ----------
    De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by
    Estimating Probability Densities. In *Advances in Neural Information
    Processing Systems* (NIPS) 9, pp. 424–430.

    Note
    ----
    MIMIC cannot be used for solving continuous-state optimization problems.
    """
    if problem.get_prob_type() == 'continuous':
        raise Exception("""problem type must be discrete or tsp.""")

    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (keep_pct < 0) or (keep_pct > 1):
        raise Exception("""keep_pct must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    if curve:
        fitness_curve = []

    if fast_mimic not in (True, False):
        raise Exception("""fast_mimic mode must be a boolean.""")
    else:
        problem.mimic_speed = fast_mimic

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Get top n percent of population
        problem.find_top_pct(keep_pct)

        # Update probability estimates
        problem.eval_node_probs()

        # Generate new sample
        new_sample = problem.sample_pop(pop_size)
        problem.set_population(new_sample)

        next_state = problem.best_child()

        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if curve:
            fitness_curve.append(max(problem.get_fitness(), next_fitness))
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state().astype(int)

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness


def run():
    start = time.time()
    best_state, best_fitness, curve = mimic(
        problems['2colors'], pop_size=2000, keep_pct=0.1, max_attempts=10, 
        max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)
    t = time.time() - start
    plot_fitness_curve('2colors', alg_name, best_state, best_fitness, curve, t)
    
    start = time.time()
    best_state, best_fitness, curve = mimic(
        problems['queens'], pop_size=10000, keep_pct=0.1, max_attempts=10, 
        max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)
    t = time.time() - start
    plot_fitness_curve('queens', alg_name, best_state, best_fitness, curve, t)
    
    start = time.time()
    best_state, best_fitness, curve = mimic(
        problems['4peaks'], pop_size=10000, keep_pct=0.1, max_attempts=10, 
        max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)
    t = time.time() - start
    plot_fitness_curve('4peaks', alg_name, best_state, best_fitness, curve, t)
    
    start = time.time()
    best_state, best_fitness, curve = mimic(
        problems['3colors'], pop_size=2000, keep_pct=0.1, max_attempts=10, 
        max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)
    t = time.time() - start
    plot_fitness_curve('3colors', alg_name, best_state, best_fitness, curve, t)
    
    start = time.time()
    best_state, best_fitness, curve = mimic(
        problems['flip'], pop_size=2000, keep_pct=0.1, max_attempts=10, 
        max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)
    t = time.time() - start
    plot_fitness_curve('flip', alg_name, best_state, best_fitness, curve, t)


if __name__ == '__main__':
    sizes = [100, 200, 500, 1000, 2000]
    pcts = [.03, .05, .07, .1, .15, .2]
    tsts = []
    for size in sizes:
        for pct in pcts:
            best_state, best_fitness, curve = mlrose.mimic( 
                problems['2colors'], pop_size=size, keep_pct=pct, max_attempts=10, 
                max_iters=50, curve=True, random_state=rand_seed, fast_mimic=True)
            print(size, pct, best_fitness)
            plot_fitness_curve('2colors', curve, alg_name)
