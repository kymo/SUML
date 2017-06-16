DIR=$(shell pwd)

MACRO=DEBUGALL
CFLAGS+=-pg -w -D$(MACRO) -lpthread
SRCPATH=.
INCLUDEPATH=${DIR}/../Feature \
			${DIR}/../Util \

ALIAS=${DIR}/../Util/Util.o \
	  ${DIR}/../Feature/Feature.o 

TARGET=alias_o test_reg_tree test_cla_tree
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

test_reg_tree:
	g++ ${CFLAGS} -o test_reg_tree test_reg_tree.o RegTree.o ${ALIAS}

test_cla_tree:
	g++ ${CFLAGS} -o test_cla_tree test_cla_tree.o ClassifyTree.o ${ALIAS}

