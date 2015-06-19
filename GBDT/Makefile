DIR=$(shell pwd)

GBDT_O=${DIR}/../Obj/GBDT.o
GBDT_SRC=${DIR}/GBDT.cpp

INC=-I ${DIR}/ -I ${DIR}/../Model -I ${DIR}/../Util -I ${DIR}/../Tree

GCC=g++

${GBDT_O}:${GBDT_SRC}
	${GCC} -c ${GBDT_SRC} -g ${INC} -o ${GBDT_O}


clean:
	rm -rf ${DIR}/../Obj/GBDT.o *.o
