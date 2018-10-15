CC = nvcc
CFLAGS = -std=c++11 -Xcompiler -fopenmp
INCLUDES = 
LDFLAGS = 
SOURCES = letter_frequency.cu
OUTF = letter_frequency
OBJS = letter_frequency.o

$(OUTF): $(OBJS)
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

$(OBJS): $(SOURCES)
	$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	rm *.o $(OUTF)