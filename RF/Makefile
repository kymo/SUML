DIR=$(shell pwd)

MACRO=DEBUGALL
CFLAGS+=-pg -w -D$(MACRO) -lpthread
SRCPATH=.
INCLUDEPATH=${DIR}/../Feature \
			${DIR}/../Util \
			${DIR}/../Model \
			${DIR}/../Tree 

ALIAS=${DIR}/../Util/Util.o \
	  ${DIR}/../Feature/Feature.o \
	  ${DIR}/../Tree/RegTree.o \
	  ${DIR}/../Tree/ClassifyTree.o 


TARGET=alias_o test_reg_rf test_cla_rf
SOURCES=$(wildcard $(SRCPATH)/*.cpp)
#OBJS := $(patsubst %.cpp, %.o,$(SOURCES))
OBJS=$(SOURCES:.cpp=.o)
CFLAGS+=$(foreach dir, $(INCLUDEPATH), -I$(dir))

all: ${TARGET}
	@echo "Build Done!"
	@echo $(CFLAGS)

clean:
	$(RM) $(OBJS) $(TARGET)

alias_o:
	g++ -c $(SOURCES) ${CFLAGS}

test_reg_rf:
	g++ ${CFLAGS} -o test_reg_rf test_reg_rf.o  RegRF.o ${ALIAS}

test_cla_rf:
	g++ ${CFLAGS} -o test_cla_rf test_cla_rf.o ClaRF.o  ${ALIAS}

