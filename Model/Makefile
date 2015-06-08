DIR=$(shell pwd)

MODEL_O=${DIR}/../Obj/Model.o
MODEL_SRC=${DIR}/Model.cpp
INC=-I ${DIR}/ -I ${DIR}/../Util

GCC=g++

${MODEL_O}:${MODEL_SRC}
	${GCC} -c ${MODEL_SRC} -g ${INC} -o ${MODEL_O}

clean:
	rm -rf *.o ${MODEL_O}
