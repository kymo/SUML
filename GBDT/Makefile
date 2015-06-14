DIR=$(shell pwd)

LR_O=${DIR}/../Obj/GBDT.o
LR_SRC=${DIR}/GBDT.cpp
INC=-I ${DIR}/ -I ${DIR}/../Model -I ${DIR}/../Util

GCC=g++

${LR_O}:${LR_SRC}
	${GCC} -c ${LR_SRC} -g ${INC} -o ${LR_O}


clean:
	rm -rf ${DIR}/../Obj/GBDT.o *.o
