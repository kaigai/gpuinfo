MODULE_big = gputest
OBJS = gputest.o
EXTRA_CLEAN = gpuinfo gpucc gpudma memeat

# Header and Libraries of OpenCL (to be autoconf?)
IPATH_LIST := /usr/include \
    /usr/local/cuda/include \
    /opt/AMDAPP/include
LPATH_LIST := /usr/lib64 \
    /usr/lib \
    /usr/local/cuda/lib64 \
    /usr/local/cuda/lib
IPATH := $(shell for x in $(IPATH_LIST);    \
           do test -e "$$x/CL/cl.h" && (echo -I $$x; break); done)
LPATH := $(shell for x in $(LPATH_LIST);    \
           do test -e "$$x/libOpenCL.so" && (echo -L $$x; break); done)
PG_CPPFLAGS := $(IPATH)
SHLIB_LINK := $(LPATH) -lcuda

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

misc: $(EXTRA_CLEAN)

gpuinfo: gpuinfo.c misc.c
	$(CC) $(CFLAGS) $^ -o $@ $(IPATH) $(LPATH) -lOpenCL

gpucc: gpucc.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl $(IPATH) $(LPATH)

gpudma: gpudma.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl $(IPATH) $(LPATH)

gpustub: gpustub.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl $(IPATH) $(LPATH)

cudadma: cudadma.c
	$(CC) $(CFLAGS) -I$(CUDA_DIR)/include $^ -o $@ -lcuda

memeat: memeat.c
	$(CC) $(CFLAGS) $^ -o $@
