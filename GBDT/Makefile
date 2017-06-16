DIR=$(shell pwd)

MACRO=DEBUGALL
CFLAGS+=-pg -w -D$(MACRO) -lpthread
SRCPATH=.
INCLUDEPATH=${DIR}/../Model \
			${DIR}/../Util \
			${DIR}/../Tree

ALIAS=${DIR}/../Feature/Feature.o \
	  ${DIR}/../Util/Util.o \
	  ${DIR}/../Tree/RegTree.o

TARGET=test_gbdt
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

