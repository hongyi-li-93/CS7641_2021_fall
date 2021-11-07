# CS7641 Assignment 3
This is for Hongyi's assignment 3.

All Python scripts are contained directly in this folder.
Subfolder data/ contains my 2 datasets, ‘abalone’ and ‘iris’ from UCI repo: https://archive.ics.uci.edu/ml/datasets.php?format=&task=cla&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=table

The python scripts will go to the data folder to read data, so make sure the current working directory is this folder. One can use os.chdir to change working directory.

data_schema.py defined schemas for both data sets and set up reading function. It's the same as previous projects. 

gap.py implements GAP statistic and test_gap is unit test.

variable_trans.py is the key library for variable transformation. It performs transformation for PAC, ICA, Kernel PCA, and it defined random components and calculate transform and inverse transform. 

kmeans.py contains algorithm to select best number of clusters and fits Kmeans.

kgauss.py contains algorithm to select best number of clusters and fits Gaussian-mixture model.

neural_net.py is the library for neural network, it fits neural network using cross-validation.

kgauss_demo.y and ica_demo.py contains toy experiment to show power of Gaussian-mixure and ICA, they are not important.

experiment.py is the main scrip to run. It read data, perform different combinations of clustering and variable transformation, and it runs neural network for Abalone data. It will create plots in the working folder with self-interpretable names as png and numerical values such as training error will be stored as txt.

All parameters to generate the report is hard coded in experiment.py as well as random seed, so just run like: 
python -m experiment
