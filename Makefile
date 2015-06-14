
GCC=g++
DIR=$(shell pwd)

INC=-I ${DIR}/LR -I ${DIR}/Model -I ${DIR}/Util -I ${DIR}/Matrix -I ${DIR}/NN

## Logistic Regression Test
TEST_LR_SRC=${DIR}/LR/test_lr.cpp
TEST_LR_O=${DIR}/Obj/test_lr.o 
TEST_LR_ALIAS_O=${DIR}/Obj/LR.o ${DIR}/Obj/Model.o ${DIR}/Obj/Util.o
TEST_LR=${DIR}/lr

## Neural Network Test
TEST_NN_SRC=${DIR}/NN/test_ann.cpp
TEST_NN_O=${DIR}/Obj/test_nn.o 
TEST_NN_ALIAS_O=${DIR}/Obj/ANN.o ${DIR}/Obj/Model.o ${DIR}/Obj/Util.o
TEST_NN=${DIR}/nn

all:${TEST_NN} ${TEST_LR}

${TEST_LR}:${TEST_LR_O}
	${GCC} -o ${TEST_LR} ${TEST_LR_O} ${TEST_LR_ALIAS_O} ${INC}

${TEST_LR_O}:${TEST_LR_SRC}
	${GCC} -c ${TEST_LR_SRC} -o ${TEST_LR_O} ${INC}


${TEST_NN}:${TEST_NN_O}
	${GCC} -o ${TEST_NN} ${TEST_NN_O} ${TEST_NN_ALIAS_O} ${INC}

${TEST_NN_O}:${TEST_NN_SRC}
	${GCC} -c ${TEST_NN_SRC} -o ${TEST_NN_O} ${INC}

clean:
	cd LR;make clean;
	cd Model;make clean;
	cd Util;make clean;
	cd Matrix;make clean;
	cd NN;make clean;
	rm -rf *.o ${TEST_LR} ${TEST_NN}

target:
	cd LR;make;
	cd Model;make;
	cd Util;make;
	cd Matrix;make;
	cd NN;make;
	make;
