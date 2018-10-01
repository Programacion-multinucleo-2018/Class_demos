CC = g++
CFLAGS = -std=c++11
INCLUDES = 
LDFLAGS = 
SOURCES = mapReduce_tests.cpp
OUTF = mapReduce_tests.exe
OBJS = mapReduce_tests.o

$(OUTF): $(OBJS)
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

$(OBJS): $(SOURCES)
	$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	rm *.o $(OUTF)