DIR=$(shell pwd)

UTIL_O=${DIR}/../Obj/Util.o
UTIL_SRC=${DIR}/Util.cpp
INC=-I ${DIR}/

GCC=g++

${UTIL_O}:${UTIL_SRC}
	${GCC} -c ${UTIL_SRC} -g ${INC} -o ${UTIL_O}

clean:
	rm -rf *.o ${UTIL_O}
