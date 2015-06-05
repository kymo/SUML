MODEL_O=Model.o
MODEL_SRC=./Model/Model.cpp
MODEL_INC=./Model

LR_O=LR.o
LR_SRC=./LR/LR.cpp
LR_INC=./LR

TEST_LR_O=test_lr.o
TEST_LR_SRC=./LR/test_lr.cpp

TEST_LR=lr

UTIL_INC=Util/
UTIL_O=Util.o
UTIL_SRC=./Util/Util.cpp
GCC=g++

${TEST_LR}:${LR_O} ${TEST_LR_O} ${MODEL_O} ${UTIL_O}
	${GCC} -o ${TEST_LR} ${LR_O} ${TEST_LR_O} ${MODEL_O} ${UTIL_O} -I ${MODEL_INC} -I ${LR_INC} -I ${UTIL_INC}

${TEST_LR_O}:${TEST_LR_SRC}
	${GCC} -c ${TEST_LR_SRC} -g  -I ${MODEL_INC} -I ${LR_INC} -I ${UTIL_INC} 





${LR_O}:${LR_SRC}
	${GCC} -c ${LR_SRC} -g -I ${LR_INC} -I ${MODEL_INC} -I ${UTIL_INC}


#	
${MODEL_O}:${MODEL_SRC}
	${GCC} -c ${MODEL_SRC} -I${MODEL_INC} -I ${UTIL_INC}

${UTIL_O}:${UTIL_SRC}
	${GCC} -c ${UTIL_SRC} -I${UTIL_INC}

clean:
	rm -rf *.o
