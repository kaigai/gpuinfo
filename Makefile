MODULE_big = gputest
OBJS = gputest.o
EXTRA_CLEAN = gpuinfo gpucc gpudma memeat nvinfo

# Header and Libraries of OpenCL (to be autoconf?)
IPATH_LIST := /usr/include \
    /usr/local/cuda/include \
    /opt/AMDAPP/include
LPATH_LIST := /usr/lib64 \
    /usr/lib \
    /usr/local/cuda/lib64 \
    /usr/local/cuda/lib
CL_IPATH := $(shell for x in $(IPATH_LIST);    \
           do test -e "$$x/CL/cl.h" && (echo -I $$x; break); done)
CL_LPATH := $(shell for x in $(LPATH_LIST);    \
           do test -e "$$x/libOpenCL.so" && (echo -L $$x; break); done)
CUDA_IPATH := $(shell for x in $(IPATH_LIST);    \
           do test -e "$$x/cuda.h" && (echo -I $$x; break); done)
CUDA_LPATH := $(shell for x in $(LPATH_LIST);    \
           do test -e "$$x/libcuda.so" && (echo -L $$x; break); done)

PG_CPPFLAGS := $(IPATH)
SHLIB_LINK := $(LPATH) -lcuda

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

misc: $(EXTRA_CLEAN)

gpuinfo: gpuinfo.c misc.c
	$(CC) $(CFLAGS) $^ -o $@ -lOpenCL $(CL_IPATH) $(CL_LPATH)

gpucc: gpucc.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl $(CL_IPATH) $(CL_LPATH)

gpudma: gpudma.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl $(CL_IPATH) $(CL_LPATH)

gpustub: gpustub.c opencl_entry.c
	$(CC) $(CFLAGS) $^ -o $@ -ldl $(CL_IPATH) $(CL_LPATH)

cudadma: cudadma.c
	$(CC) $(CFLAGS) $^ -o $@ -lcuda $(CUDA_IPATH) $(CUDA_LPATH)

nvinfo: nvinfo.c
	$(CC) $(CFLAGS) $^ -o $@ -lcuda $(CUDA_IPATH) $(CUDA_LPATH)

memeat: memeat.c
	$(CC) $(CFLAGS) $^ -o $@
