GCC=g++
DIR=$(shell pwd)

INC=-I ${DIR}/LR \
	-I ${DIR}/Model\
   	-I ${DIR}/Util \
	-I ${DIR}/Matrix\
   	-I ${DIR}/NN\
	-I ${DIR}/Tree \
	-I ${DIR}/Feature \
	-I ${DIR}/RF \
	-I ${DIR}/SVM \
	-g

## Logistic Regression Test
TEST_LR_SRC=${DIR}/LR/test_lr.cpp
TEST_LR_O=${DIR}/Obj/test_lr.o 
TEST_LR_ALIAS_O=${DIR}/Obj/LR.o ${DIR}/Obj/Util.o ${DIR}/Obj/Feature.o
TEST_LR=${DIR}/Bin/lr

## Neural Network Test
TEST_NN_SRC=${DIR}/NN/test_ann.cpp 
TEST_NN_O=${DIR}/Obj/test_nn.o 
TEST_NN_ALIAS_O=${DIR}/Obj/ANN.o ${DIR}/Obj/Util.o ${DIR}/Obj/Feature.o
TEST_NN=${DIR}/Bin/nn

## regression Tree Test (variance)
TEST_REG_TREE_SRC=${DIR}/Tree/test_reg_tree.cpp
TEST_REG_TREE_O=${DIR}/Obj/test_reg_tree.o
TEST_REG_TREE_ALIAS_O=${DIR}/Obj/RegTree.o ${DIR}/Obj/Util.o -lpthread ${DIR}/Obj/Feature.o
TEST_REG_TREE=${DIR}/Bin/reg_tree

## classification Tree Test (gini)
TEST_CAL_TREE_SRC=${DIR}/Tree/test_cla_tree.cpp
TEST_CAL_TREE_O=${DIR}/Obj/test_cla_tree.o
TEST_CAL_TREE_ALIAS_O=${DIR}/Obj/ClassifyTree.o ${DIR}/Obj/Util.o -lpthread ${DIR}/Obj/Feature.o
TEST_CAL_TREE=${DIR}/Bin/cla_tree

## gradient boosting decision tree test
TEST_GBDT_SRC=${DIR}/GBDT/test_gbdt.cpp
TEST_GBDT_O=${DIR}/Obj/test_gbdt.o
TEST_GBDT_ALIAS_O=${DIR}/Obj/RegTree.o ${DIR}/Obj/Util.o -lpthread ${DIR}/Obj/Feature.o ${DIR}/Obj/GBDT.o
TEST_GBDT=${DIR}/Bin/gbdt

## random forest regressor
TEST_RF_REG_SRC=${DIR}/RF/test_reg_rf.cpp
TEST_RF_REG_O=${DIR}/Obj/test_reg_rf.o
TEST_RF_REG_ALIAS_O=${DIR}/Obj/RegRF.o ${DIR}/Obj/Util.o -lpthread ${DIR}/Obj/Feature.o ${DIR}/Obj/RegTree.o
TEST_RF_REG=${DIR}/Bin/rf_reg

## random forest classifier
TEST_RF_CLA_SRC=${DIR}/RF/test_cla_rf.cpp
TEST_RF_CLA_O=${DIR}/Obj/test_cla_rf.o
TEST_RF_CLA_ALIAS_O=${DIR}/Obj/ClaRF.o ${DIR}/Obj/Util.o -lpthread ${DIR}/Obj/Feature.o ${DIR}/Obj/ClassifyTree.o
TEST_RF_CLA=${DIR}/Bin/rf_cla

## support vector machine
TEST_SVM_SRC=${DIR}/SVM/test_svm.cpp
TEST_SVM_O=${DIR}/Obj/test_svm.o
TEST_SVM_ALIAS_O=${DIR}/Obj/SVM.o ${DIR}/Obj/Util.o -lpthread ${DIR}/Obj/Feature.o
TEST_SVM=${DIR}/Bin/svm

all:${TEST_NN} \
	${TEST_LR} \
	${TEST_CAL_TREE} \
	${TEST_REG_TREE} \
	${TEST_CAL_TREE} \
	${TEST_GBDT} \
	${TEST_RF_REG} \
	${TEST_RF_CLA} \
	${TEST_SVM}

.PHONY : all target clean

# generate LR test runable program
${TEST_LR}:${TEST_LR_O}
	${GCC} -o ${TEST_LR} ${TEST_LR_O} ${TEST_LR_ALIAS_O} ${INC}

${TEST_LR_O}:${TEST_LR_SRC}
	${GCC} -c ${TEST_LR_SRC} -o ${TEST_LR_O} ${INC}

# generate neural network runable program
${TEST_NN}:${TEST_NN_O}
	${GCC} -o ${TEST_NN} ${TEST_NN_O} ${TEST_NN_ALIAS_O} ${INC}

${TEST_NN_O}:${TEST_NN_SRC}
	${GCC} -c ${TEST_NN_SRC} -o ${TEST_NN_O} ${INC}

# generate regression tree runable program
${TEST_REG_TREE}:${TEST_REG_TREE_O}
	${GCC} -o ${TEST_REG_TREE} ${TEST_REG_TREE_O} ${TEST_REG_TREE_ALIAS_O} ${INC}

${TEST_REG_TREE_O}:${TEST_REG_TREE_SRC}
	${GCC} -c ${TEST_REG_TREE_SRC} -o ${TEST_REG_TREE_O} ${INC}

# generate classification tree runable program
${TEST_CAL_TREE}:${TEST_CAL_TREE_O}
	${GCC} -o ${TEST_CAL_TREE} ${TEST_CAL_TREE_O} ${TEST_CAL_TREE_ALIAS_O} ${INC}

${TEST_CAL_TREE_O}:${TEST_CAL_TREE_SRC}
	${GCC} -c ${TEST_CAL_TREE_SRC} -o ${TEST_CAL_TREE_O} ${INC}

# generate gbdt runable program
${TEST_GBDT}:${TEST_GBDT_O}
	${GCC} -o ${TEST_GBDT} ${TEST_GBDT_O} ${TEST_GBDT_ALIAS_O} ${INC}

${TEST_GBDT_O}:${TEST_GBDT_SRC}
	${GCC} -c ${TEST_GBDT_SRC} -o ${TEST_GBDT_O} ${INC}

# generate random forest regression runable program
${TEST_RF_REG}:${TEST_RF_REG_O}
	${GCC} -o ${TEST_RF_REG} ${TEST_RF_REG_O} ${TEST_RF_REG_ALIAS_O} ${INC}

${TEST_RF_REG_O}:${TEST_RF_REG_SRC}
	${GCC} -c ${TEST_RF_REG_SRC} -o ${TEST_RF_REG_O} ${INC}

# generate random forest regression runable program
${TEST_RF_CLA}:${TEST_RF_CLA_O}
	${GCC} -o ${TEST_RF_CLA} ${TEST_RF_CLA_O} ${TEST_RF_CLA_ALIAS_O} ${INC}

${TEST_RF_CLA_O}:${TEST_RF_CLA_SRC}
	${GCC} -c ${TEST_RF_CLA_SRC} -o ${TEST_RF_CLA_O} ${INC}

# generate support vector machine runable program
${TEST_SVM}:${TEST_SVM_O}
	${GCC} -o ${TEST_SVM} ${TEST_SVM_O} ${TEST_SVM_ALIAS_O} ${INC}

${TEST_SVM_O}:${TEST_SVM_SRC}
	${GCC} -c ${TEST_SVM_SRC} -o ${TEST_SVM_O} ${INC}

clean:
	cd LR;make clean;
	cd Util;make clean;
	cd Matrix;make clean;
	cd NN;make clean;
	cd Feature;make clean;
	cd Tree;make clean;
	cd GBDT;make clean;
	cd RF;make clean;
	cd SVM;make clean;
	rm -rf *.o ${TEST_LR} ${TEST_NN} ${TEST_REG_TREE} ${TEST_CAL_TREE}
	rm -rf ${DIR}/Obj/*.o
	rm -rf ${DIR}/Bin/*

target:
	cd LR;make;
	cd Util;make;
	cd Feature;make;
	cd Matrix;make;
	cd NN;make;
	cd Tree;make;
	cd GBDT;make;
	cd RF;make;
	cd SVM;make;
	make;
