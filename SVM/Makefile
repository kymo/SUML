CC=g++
CFLAGS=-g -Wall
exe=svm
obj=svm.o main.o
$(exe):$(obj)
	$(CC) -o $(exe) $(obj)
main.o:main.cpp
	$(CC) $(CFLAGS) -c main.cpp
svm.o:svm.cpp
	$(CC) $(CFLAGS) -c svm.cpp


clean:
	rm *.o svm
