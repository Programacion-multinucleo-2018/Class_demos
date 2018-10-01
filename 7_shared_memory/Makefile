CC = nvcc
CFLAGS = -Wno-deprecated-gpu-targets -std=c++11
INCLUDES = 
LDFLAGS = 
SOURCES = shared_memory.cu
OUTF = shared_memory.exe
OBJS = shared_memory.o

$(OUTF): $(OBJS)
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

$(OBJS): $(SOURCES)
	$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	rm *.o $(OUTF)