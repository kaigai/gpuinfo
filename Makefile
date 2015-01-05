MODULES = gputest
EXTRA_CLEAN = gpuinfo gpucc gpudma memeat

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
