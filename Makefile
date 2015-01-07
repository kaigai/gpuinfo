MODULE_big = gputest
OBJS = gputest.o
EXTRA_CLEAN = gpuinfo gpucc gpudma memeat
PG_CPPFLAGS := -I/usr/local/cuda/include
SHLIB_LINK := -L/usr/local/cuda/lib64 -lcuda

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

misc: $(EXTRA_CLEAN)

gpuinfo: gpuinfo.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl

gpucc: gpucc.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl

gpudma: gpudma.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl

gpustub: gpustub.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl

cudadma: cudadma.c
	$(CC) $(CFLAGS) -I$(CUDA_DIR)/include $^ -o $@ -lcuda

memeat: memeat.c
	$(CC) $(CFLAGS) $^ -o $@
