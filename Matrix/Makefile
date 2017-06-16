DIR=$(shell pwd)

MACRO=DEBUGALL
CFLAGS+=-pg -w -D$(MACRO)
SRCPATH=.
INCLUDEPATH=${DIR}/../Util 
ALIAS=${DIR}/../Util/Util.o 

TARGET=test_matrix
SOURCES=$(wildcard $(SRCPATH)/*.cpp)

#OBJS := $(patsubst %.cpp, %.o,$(SOURCES))
OBJS=$(SOURCES:.cpp=.o)
CFLAGS+=$(foreach dir, $(INCLUDEPATH), -I$(dir))

all: build
	@echo "Build Done!"
	@echo $(CFLAGS)
clean:
	$(RM) $(OBJS) $(TARGET)

build:
	g++ -c ${CFLAGS} ${SOURCES}
	g++ ${CFLAGS} -o ${TARGET} ${OBJS} ${ALIAS}

