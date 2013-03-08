gpuinfo: gpuinfo.c opencl_entry.c
	$(CC) $^ -o $@ -ldl
