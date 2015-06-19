
DIR=$(shell pwd)

RegTree_O=${DIR}/../Obj/RegTree.o
RegTree_SRC=${DIR}/RegTree.cpp

ClassifyTree_O=${DIR}/../Obj/ClassifyTree.o
ClassifyTree_SRC=${DIR}/ClassifyTree.cpp

INC=-I ${DIR}/ \
	-I ${DIR}/../Util

GCC=g++

all: ${RegTree_O} ${ClassifyTree_O}

${RegTree_O}:${LR_SRC}
	${GCC} -c ${RegTree_SRC} ${INC} -o ${RegTree_O}

${ClassifyTree_O}:${ClassifyTree_SRC}
	${GCC} -c ${ClassifyTree_SRC} ${INC} -o ${ClassifyTree_O}

clean:
	rm -rf ${RegTree_O} ${ClassifyTree_O}
