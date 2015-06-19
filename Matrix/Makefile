
DIR=$(shell pwd)
OBJ=${DIR}/../Obj/Matrix.o
SRC=${DIR}/Matrix.cpp

CFLAGS=-g -I ${DIR}

GCC=g++

${OBJ}:${SRC}
	${GCC} -c ${SRC} ${CFLAGS} -o ${OBJ}

clean:
	rm -rf *.o ${OBJ}
