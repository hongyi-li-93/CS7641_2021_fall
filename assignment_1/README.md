# CS7641 Assignment 1
This is for Hongyi's assignment 1.

All Python scripts are contained directly in this folder.
Subfolder data/ contains my 2 datasets, ‘abalone’ and ‘bank-additional’ from UCI repo: https://archive.ics.uci.edu/ml/datasets.php?format=&task=cla&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=table

The python scripts will go to the data folder to read data, so make sure the current working directory is this folder. One can use os.chdir to change working directory.

data_schema.py defined schemas for both data sets and set up reading function. 
I defined to data type (class) for attributes, Numerical and Categorical while for Categorical, support has to be provided. One can defined the schema for a data set as the DataSetSchema class. Variable Data_Set_Schemas contains schemas for both data sets. Data_Set_Specs contains other specifications for the 2 data sets like suffix of data file, index of training samples and testing samples. Function get_train_test_ml_set takes data set name as input and returns a pair of formatted class MLDateSet for traning and testing samples. MLDateSet is a clean format directly usable by other scripts, it has dummy variables for categorical attributes already created. Finally, we can re-order input data to generate randomness by setting the seed.

decision_tree.py selects parameter for decision tree using cross validation and returns training / testing errors for both data sets.
One can directly run “python -m decision_tree” and cv plots and erros will be saved to current folder.

neural_net.py selects parameter for neural network using cross validation and returns training / testing errors for both data sets.
One can directly run “python -m neural_net” and cv tables and erros will be saved to current folder.

k_nearest_neighbor.py implements k nearest neighbor classifier and selects parameter of it using cross validation and returns training / testing errors for both data sets.
One can directly run “python -m k_nearest_neighbor” and cv plots and erros will be saved to current folder.

ada_boosting.py selects parameter for ada boosting using cross validation and returns training / testing errors for both data sets.
One can directly run “python -m ada_boosting” and cv plots and erros will be saved to current folder.

svm_rbf_poly.py selects parameter for SVM with rbf and poly kernels using cross validation and returns training / testing errors for both data sets.
One can directly run “python -m svm_rbf_poly” and cv plots and erros will be saved to current folder.
