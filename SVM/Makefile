DIR=$(shell pwd)

SVM_O=${DIR}/../Obj/SVM.o
SVM_SRC=${DIR}/SVM.cpp
INC=-I ${DIR}/ -I ${DIR}/../Model -I ${DIR}/../Util

GCC=g++

${SVM_O}:${SVM_SRC}
	${GCC} -c ${SVM_SRC} -g ${INC} -o ${SVM_O}


clean:
	rm -rf ${DIR}/../Obj/SVM.o *.o
