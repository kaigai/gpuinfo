all: gpuinfo gpucc gpudma

CFLAGS := -g -O2

gpuinfo: gpuinfo.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl

gpucc: gpucc.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl

gpudma: gpudma.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl

clean:
	rm -f gpuinfo gpucc
