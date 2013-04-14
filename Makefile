all: gpuinfo gpucc

gpuinfo: gpuinfo.c opencl_entry.c
	$(CC) $^ -o $@ -ldl

gpucc: gpucc.c opencl_entry.c
	$(CC) $^ -o $@ -ldl

clean:
	rm -f gpuinfo gpucc
