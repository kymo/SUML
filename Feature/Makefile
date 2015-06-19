DIR=$(shell pwd)

Feature_O=${DIR}/../Obj/Feature.o
Feature_SRC=${DIR}/Feature.cpp
INC=-I ${DIR}/ \
	-I ${DIR}/../Util
GCC=g++

${Feature_O}:${Feature_SRC}
	${GCC} -c ${Feature_SRC} -g ${INC} -o ${Feature_O}


clean:
	rm -rf ${DIR}/../Obj/Feature.o *.o
