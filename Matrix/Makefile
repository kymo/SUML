
DIR=$(shell pwd)
OBJ=${DIR}/../Obj/Matrix.o
SRC=${DIR}/Matrix.cpp

CFLAGS=-g -I ${DIR} -I ${DIR}/../Util
GCC=g++

${OBJ}:${SRC}
	${GCC} -c ${SRC} ${CFLAGS} -o ${OBJ}

clean:
	rm -rf *.o ${OBJ}
