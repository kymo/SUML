# SUML
* SUML is a machine learning basic source code written in c plus plus, and currently svm, random forest, gradient boosting descent tree, logistic regression and artificial neural network have been implemented in current version, and the optimization process in on the go.
* next I will add GMM and ME, maybe some kind of feature enginnering method will be added into this project too.

# Makefile
* main makefile shows up in the root directory, and the makefiles file for each module are included in each child directory, just run "make target" you can get the test runable program

# Module
* Support Vector Machine : optimized by SMO
* Random Forests : just for regression, classification will be added soon
* Gradient Boosting Descent Tree : needs to be optimized
* Logistic Regression : sgd & gd, l1 & l2 regularization
* Nueral Network : neural network optimized by stochastic gradient descent, shuffle samples will be add.
* Model : basic model for these algorithms, supply the basic operation of feature.
* Matrix : implementation for matrix operation
* Util : common tools just like sort, split_string and so on.
