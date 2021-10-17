#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 19:27:34 2021

@author: lihongyi
"""


import numpy as np
import time
from problems import rand_seed, problems, plot_fitness_curve, plt
from data_schema import get_train_test_ml_set
from neural_net import train_network, error_rate, cv_err


alg_name = 'genetic alg'


def genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10,
                max_iters=np.inf, curve=False, random_state=None, threshold=None):
    """Use a standard genetic algorithm to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    pop_size: int, default: 200
        Size of population to be used in genetic algorithm.
    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector
        during reproduction, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
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

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.
    fitness_curve: array
        Numpy array of arrays containing the fitness of the entire population
        at every iteration.
        Only returned if input argument :code:`curve` is :code:`True`.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (mutation_prob < 0) or (mutation_prob > 1):
        raise Exception("""mutation_prob must be between 0 and 1.""")

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

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Calculate breeding probabilities
        if threshold is not None:
            problem.eval_mate_probs_threshold(prob=threshold)
        else:
            problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []

        for _ in range(pop_size):
            # Select parents
            selected = np.random.choice(pop_size, size=2,
                                        p=problem.get_mate_probs())
            parent_1 = problem.get_population()[selected[0]]
            parent_2 = problem.get_population()[selected[1]]

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        next_gen = np.array(next_gen)
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)
        
        print(iters, next_fitness)

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
    best_state = problem.get_state()

    if curve:
        return best_state, best_fitness, np.asarray(fitness_curve)

    return best_state, best_fitness


def run():
    start = time.time()
    best_state, best_fitness, curve = genetic_alg(
        problems['queens'], pop_size=10000, mutation_prob=0.2, max_attempts=10, 
        max_iters=100, curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('queens', alg_name, best_state, best_fitness, curve, t)

    start = time.time()
    best_state, best_fitness, curve = genetic_alg(
        problems['4peaks'], pop_size=50000, mutation_prob=0.2, max_attempts=10, 
        max_iters=100, curve=True, random_state=rand_seed, threshold=.95)
    t = time.time() - start
    plot_fitness_curve('4peaks', alg_name, best_state, best_fitness, curve, t)

    start = time.time()
    best_state, best_fitness, curve = genetic_alg(
        problems['3colors'], pop_size=10000, mutation_prob=0.2, max_attempts=10, 
        max_iters=100, curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('3colors', alg_name, best_state, best_fitness, curve, t)

    start = time.time()
    best_state, best_fitness, curve = genetic_alg(
        problems['flip'], pop_size=10000, mutation_prob=0.2, max_attempts=10, 
        max_iters=100, curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('flip', alg_name, best_state, best_fitness, curve, t)
    
    start = time.time()
    best_state, best_fitness, curve = genetic_alg(
        problems['2colors'], pop_size=10000, mutation_prob=0.2, max_attempts=10, 
        max_iters=100, curve=True, random_state=rand_seed)
    t = time.time() - start
    plot_fitness_curve('2colors', alg_name, best_state, best_fitness, curve, t)


def nn():
    algorithm = 'genetic_alg'
    train_set, test_set = get_train_test_ml_set('abalone')
    max_iters=500
    step_size=.02
    pop_mut = ['pop 200 prob 0.1', 'pop 800 prob 0.1', 'pop 200 prob 0.2', 'pop 800 prob 0.2']
    settings = {
        'pop 200 prob 0.1': (200, .1),
        'pop 800 prob 0.1': (800, .1),
        'pop 200 prob 0.2': (200, .2),
        'pop 800 prob 0.2': (800, .2),
        }
    
    errs = []
    costs = []
    curves = []
    for setting in pop_mut:
        pop_size, mutation_prob = settings[setting]
        err, t_cost, curve = cv_err(
            train_set, algorithm, step_size, 
            max_iters=max_iters, pop_size=pop_size, mutation_prob=mutation_prob)
        errs.append(err)
        costs.append(t_cost)
        curves.append(curve)
    
    for i, curve in enumerate(curves):
        plt.plot(curve, label=f'{pop_mut[i]}')
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
    
    pop_size, mutation_prob = settings[pop_mut[np.argmin(errs)]]
    scaler, clf, t, curve = train_network(
        train_set.independent_matrix, train_set.dependent_vector, 
        algorithm, step_size, 
        max_iters=max_iters, pop_size=pop_size, mutation_prob=mutation_prob)
    
    train_err = error_rate(train_set.independent_matrix, train_set.dependent_vector, scaler, clf)
    test_err = error_rate(test_set.independent_matrix, test_set.dependent_vector, scaler, clf)
    with open(f'nn_{algorithm}.txt', "w") as text_file: 
        text_file.write(str(t))
        text_file.write('\n')
        text_file.write(str(train_err))
        text_file.write('\n')
        text_file.write(str(test_err))
