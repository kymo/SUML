
GCC=g++
DIR=$(shell pwd)
TEST_LR_SRC=${DIR}/LR/test_lr.cpp
TEST_LR_O=${DIR}/Obj/test_lr.o ${DIR}/Obj/LR.o ${DIR}/Obj/Model.o ${DIR}/Obj/Util.o

TEST_LR=${DIR}/lr
INC=-I ${DIR}/LR -I ${DIR}/Model -I ${DIR}/Util -I ${DIR}/Matrix

${TEST_LR}:${TEST_LR_O}
	${GCC} -o ${TEST_LR} ${TEST_LR_O} ${INC}

${TEST_LR_O}:${TEST_LR_SRC}
	${GCC} -c ${TEST_LR_SRC} -o ${TEST_LR_O} ${INC}


clean:
	cd LR;make clean;
	cd Model;make clean;
	cd Util;make clean;
	cd Matrix;make clean;

all:
	cd LR;make;
	cd Model;make;
	cd Util;make;
	cd Matrix;make;
	make;
