# CS7641 Assignment 2
This is for Hongyi's assignment 2.

All Python scripts are contained directly in this folder.
Subfolder data/ contains my dataset for ‘abalone’ from UCI repo: https://archive.ics.uci.edu/ml/datasets.php?format=&task=cla&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=table
(The python scripts will go to the data folder to read data, so make sure the current working directory is this folder. One can use os.chdir to change working directory.)

All randomized optimization algorithms are from mlrose: https://mlrose.readthedocs.io/en/stable/source/intro.html

problems.py defines the 5 problems created by me. 
It also contains helper functions for example to plot fitness curve, and my self-defined filtering for genetic algorithm.

rhc.py sets up randomized hill climbing to fit my 5 problems and to train neural network.
The parameter setting are hard coded in this file. I modifed the curve to show attempts.

sim_anneal.py sets up simulated annealing to fit my 5 problems and to train neural network.
The parameter setting are hard coded in this file. Both geometric and arithmetic decay are included.

genetic.py sets up genetic algorith to fit my 5 problems and to train neural network.
The parameter setting are hard coded in this file. I modifed the curve to show attempts and to call my filtering.

mimic.py sets up MIMIC to fit my 5 problems.
The parameter setting are hard coded in this file. I modifed the curve to show attempts.
be carefull that MIMIC taks lots of memory to run.

data_schema.py reads from data folder for neural network training.
neural_net.py sets up wrapper to train nural network using randomized optimizations, and sets up cross validation.

run_optimization.py will run my problems for all optimization algorithms as well as train neural networks agains them.
It will save fitness curves to this folder and time cost, fitness, error and other informatons as txt files also to this folder.

to run everything, just call run_optimization.py e.g.:
python -m run_optimization.py 
