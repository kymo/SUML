DIR=$(shell pwd)

RF_O=${DIR}/../Obj/RF.o
RF_SRC=${DIR}/RF.cpp
INC=-I ${DIR}/ -I ${DIR}/../Model -I ${DIR}/../Util -I ${DIR}/../Tree -I ${DIR}/../Feature

GCC=g++

${RF_O}:${RF_SRC}
	${GCC} -c ${RF_SRC} -g ${INC} -o ${RF_O}


clean:
	rm -rf ${DIR}/../Obj/RF.o *.o
