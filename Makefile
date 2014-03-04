all: memeat gpuinfo gpucc gpudma

CFLAGS := -g -O2
CUDA_DIR := /usr/local/cuda

gpuinfo: gpuinfo.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl

gpucc: gpucc.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl

gpudma: gpudma.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl

cudadma: cudadma.c
	$(CC) $(CFLAGS) -I$(CUDA_DIR)/include $^ -o $@ -lcuda

memeat: memeat.c
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -f gpuinfo gpucc gpudma cudadma memeat
