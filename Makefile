GCC=g++
DIR=$(shell pwd)

INC=-I ${DIR}/LR \
	-I ${DIR}/Model\
   	-I ${DIR}/Util \
	-I ${DIR}/Matrix\
   	-I ${DIR}/NN\
	-I ${DIR}/Tree \
	-I ${DIR}/Feature 

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


all:${TEST_NN} ${TEST_LR} ${TEST_REG_TREE} ${TEST_CAL_TREE} ${TEST_GBDT}

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

clean:
	cd LR;make clean;
	cd Util;make clean;
	cd Matrix;make clean;
	cd NN;make clean;
	cd Feature;make clean;
	cd Tree;make clean;
	cd GBDT;make clean;
	rm -rf *.o ${TEST_LR} ${TEST_NN} ${TEST_REG_TREE} ${TEST_CAL_TREE}
	rm -rf ${DIR}/Obj/*.o

target:
	cd LR;make;
	cd Util;make;
	cd Feature;make;
	cd Matrix;make;
	cd NN;make;
	cd Tree;make;
	cd GBDT;make;
	make;
