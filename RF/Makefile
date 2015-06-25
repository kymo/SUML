DIR=$(shell pwd)

RegRF_O=${DIR}/../Obj/RegRF.o
RegRF_SRC=${DIR}/RegRF.cpp
ClaRF_O=${DIR}/../Obj/ClaRF.o
ClaRF_SRC=${DIR}/ClaRF.cpp


INC=-I ${DIR}/ -I ${DIR}/../Model -I ${DIR}/../Util -I ${DIR}/../Tree -I ${DIR}/../Feature

GCC=g++

all: ${RegRF_O} ${ClaRF_O}

${RegRF_O}:${RegRF_SRC}
	${GCC} -c ${RegRF_SRC} -g ${INC} -o ${RegRF_O}

${ClaRF_O}:${ClaRF_SRC}
	${GCC} -c ${ClaRF_SRC} -g ${INC} -o ${ClaRF_O}

clean:
	rm -rf ${ClaRF_O} ${RegRF_O} *.o
