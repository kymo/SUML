### SUML
* SUML is a machine learning library written in c++, and currently, support vector machine, random forest, gradient boosting descent tree, logistic regression, artificial neural network, conditional random fields have been implemented in version 0.1, and the optimization process in on the go.
* TODO: leverate parameter server to parallel these algorithms.

### Make
* main makefile shows up in the root directory, and the makefile for each module is included in each child directory, just run "make" in the root directory you can get all the test runable programs

# Module
* Support Vector Machine(SMO)
* Random Forests(Classification & Regression, Multithread-enabled)
* Gradient Boosting Descent Tree(Regression & Classification, Multithread-enabled)
* Logistic Regression(SGD,GD, l1-reg, l2-reg)
* CRF(conditional random fields)
* Nueral Network(Common BPNN, optimized by SGD)
* Tree(basic tree model for random forests and gbdt, and the single tree can also be used to act as a single model)
* Mode(basic model for these algorithms, supply the basic operation of feature)
* Matrix(implementation for matrix operation)
* Util(common tools just like sort, split_string and so on.)

# Test
* test logistic regression
** ./Bin/lr DATA/ionosphere 0.05 500 2 0 2 0.2
* test neural network
** ./Bin/nn DATA/ionosphere 0.05 500 10 2 1 
* test gradient boosting descent tree
** ./Bin/gbdt DATA/data 5 20 100 5 0.04 0 
* test classification tree
** ./Bin/cla_tree DATA/ionosphere 100 1000 0 0-33:6 
* test regression tree
** ./Bin/reg_tree DATA/data 20 200 0 
* test classification random forest
** ./Bin/rf_cla DATA/data 100 20 500 5 0 2 0-1:6
* test regression random forest
** ./Bin/rf_reg DATA/housing.data 100 4 20 5 0
* test support vector machine
** ./Bin/svm DATA/testSet.txt 0.1 500

