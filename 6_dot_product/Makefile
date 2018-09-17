CC = nvcc
CFLAGS = -Wno-deprecated-gpu-targets
INCLUDES = 
LDFLAGS = 
SOURCES = dot_product_cuda.cu
OUTF = dot_product_cuda.exe
OBJS = dot_product_cuda.o

$(OUTF): $(OBJS)
	$(CC) $(CFLAGS) -o $(OUTF) $< $(LDFLAGS)

$(OBJS): $(SOURCES)
	$(CC) $(CFLAGS) -c $<

rebuild: clean $(OUTF)

clean:
	rm *.o $(OUTF)