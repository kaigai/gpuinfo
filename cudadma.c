/*
 * cudadma - test for DMA transfer on CUDA device
 */
#include <errno.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda.h>

#define lengthof(array) (sizeof (array) / sizeof ((array)[0]))
#define error_exit(fmt,...)					\
	do {									\
		fprintf(stderr, "%s:%d " fmt "\n",	\
				__FUNCTION__, __LINE__,		\
				##__VA_ARGS__);				\
		exit(1);							\
	} while(0)

static int		is_blocking = 1;
static int		num_trial = 100;			/* 100 times */
static size_t	buffer_size = 128 << 20;	/* 128MB */
static size_t	chunk_size = 0;

static const char *
cuGetErrorString(CUresult errcode)
{
	static char     strbuf[256];

	switch (errcode)
	{
		case CUDA_SUCCESS:
			return "success";
		case CUDA_ERROR_INVALID_VALUE:
			return "invalid value";
		case CUDA_ERROR_OUT_OF_MEMORY:
			return "out of memory";
		case CUDA_ERROR_NOT_INITIALIZED:
			return "not initialized";
		case CUDA_ERROR_DEINITIALIZED:
			return "deinitialized";
		case CUDA_ERROR_PROFILER_DISABLED:
			return "profiler disabled";
		case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
			return "profiler not initialized";
		case CUDA_ERROR_PROFILER_ALREADY_STARTED:
			return "profiler already started";
		case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
			return "profiler already stopped";
		case CUDA_ERROR_NO_DEVICE:
			return "no device";
		case CUDA_ERROR_INVALID_DEVICE:
			return "invalid device";
		case CUDA_ERROR_INVALID_IMAGE:
			return "invalid image";
		case CUDA_ERROR_INVALID_CONTEXT:
			return "invalid context";
		case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
			return "context already current";
		case CUDA_ERROR_MAP_FAILED:
			return "map failed";
		case CUDA_ERROR_UNMAP_FAILED:
			return "unmap failed";
		case CUDA_ERROR_ARRAY_IS_MAPPED:
			return "array is mapped";
		case CUDA_ERROR_ALREADY_MAPPED:
			return "already mapped";
		case CUDA_ERROR_NO_BINARY_FOR_GPU:
			return "no binary for gpu";
		case CUDA_ERROR_ALREADY_ACQUIRED:
			return "already acquired";
		case CUDA_ERROR_NOT_MAPPED:
			return "not mapped";
		case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
			return "not mapped as array";
		case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
			return "not mapped as pointer";
		case CUDA_ERROR_ECC_UNCORRECTABLE:
			return "ecc uncorrectable";
		case CUDA_ERROR_UNSUPPORTED_LIMIT:
			return "unsupported limit";
		case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
			return "context already in use";
		case CUDA_ERROR_INVALID_SOURCE:
			return "invalid source";
		case CUDA_ERROR_FILE_NOT_FOUND:
			return "file not found";
		case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
			return "shared object symbol not found";
		case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
			return "shared object init failed";
		case CUDA_ERROR_OPERATING_SYSTEM:
			return "operating system";
		case CUDA_ERROR_INVALID_HANDLE:
			return "invalid handle";
		case CUDA_ERROR_NOT_FOUND:
			return "not found";
		case CUDA_ERROR_NOT_READY:
			return "not ready";
		case CUDA_ERROR_LAUNCH_FAILED:
			return "launch failed";
		case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
			return "launch out of resources";
		case CUDA_ERROR_LAUNCH_TIMEOUT:
			return "launch timeout";
		case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
			return "launch incompatible texturing";
		case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
			return "peer access already enabled";
		case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
			return "peer access not enabled";
		case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
			return "primary context active";
		case CUDA_ERROR_CONTEXT_IS_DESTROYED:
			return "context is destroyed";
		default:
			snprintf(strbuf, sizeof(strbuf), "cuda error = %d", errcode);
			break;
	}
	return strbuf;
}


static void
run_test(const char *namebuf, CUcontext context, CUstream stream)
{
	char	   *hmem;
	CUdeviceptr	dmem;
	int			num_chunks = buffer_size / chunk_size;
	int			i, j, k;
	CUresult	rc;
	struct timeval tv1, tv2;

	if (is_blocking)
	{
		hmem = malloc(buffer_size);
		if (!hmem)
			error_exit("failed on malloc : %s", strerror(rc));
	}
	else
	{
		rc = cuMemAllocHost((void **)&hmem, buffer_size);
		if (rc != CUDA_SUCCESS)
			error_exit("failed on cuMemAllocHost : %s", cuGetErrorString(rc));
	}
	rc = cuMemAlloc(&dmem, buffer_size);
	if (rc != CUDA_SUCCESS)
		error_exit("failed on cuMemAlloc : %s", cuGetErrorString(rc));

	gettimeofday(&tv1, NULL);
	for (i=0, k=0; i < num_trial; i++)
	{
		for (j=0; j < num_chunks; j++)
		{
			if (is_blocking)
			{
				rc = cuMemcpyHtoD(dmem + j * chunk_size,
								  hmem + j * chunk_size,
								  chunk_size);
				if (rc != CUDA_SUCCESS)
					error_exit("failed on cuMemcpyHtoD : %s",
							   cuGetErrorString(rc));
			}
			else
			{
				rc = cuMemcpyHtoDAsync(dmem + j * chunk_size,
									   hmem + j * chunk_size,
									   chunk_size,
									   stream);
				if (rc != CUDA_SUCCESS)
					error_exit("failed on cuMemcpyHtoDAsync : %s",
                               cuGetErrorString(rc));
			}
		}

		if (is_blocking)
		{
			rc = cuMemcpyDtoH(hmem, dmem, buffer_size);
			if (rc != CUDA_SUCCESS)
				error_exit("failed on cuMemcpyDtoH : %s",
						   cuGetErrorString(rc));
		}
		else
		{
			rc = cuMemcpyDtoHAsync(hmem, dmem, buffer_size, stream);
			if (rc != CUDA_SUCCESS)
                error_exit("failed on cuMemcpyDtoHAsync : %s",
                           cuGetErrorString(rc));
		}
	}
	/* wait for completion */
	rc = cuCtxSynchronize();
	if (rc != CUDA_SUCCESS)
		error_exit("failed on cuCtxSynchronize : %s", cuGetErrorString(rc));

	gettimeofday(&tv2, NULL);

	printf("DMA send/recv test result\n"
		   "device:         %s\n"
		   "size:           %luMB\n"
		   "chunks:         %lu%s x %d\n"
		   "ntrials:        %d\n"
		   "total_size:     %luMB\n"
		   "time:           %.2fs\n"
		   "speed:          %.2fMB/s\n"
		   "mode:           %s\n",
		   namebuf,
		   buffer_size >> 20,
		   chunk_size > (1UL<<20) ? chunk_size >> 20 : chunk_size >> 10,
		   chunk_size > (1UL<<20) ? "MB" : "KB",
		   num_chunks,
		   num_trial,
		   (buffer_size >> 20) * num_trial,
		   (double)((tv2.tv_sec * 1000000 + tv2.tv_usec) -
					(tv1.tv_sec * 1000000 + tv1.tv_usec)) / 1000000.0,
		   (double)(((buffer_size >> 20) * num_trial) * 1000000) /
           (double)((tv2.tv_sec * 1000000 + tv2.tv_usec) -
                    (tv1.tv_sec * 1000000 + tv1.tv_usec)),
		   is_blocking ? "sync" : "async");
	/* release resources */
	cuMemFree(dmem);
	cuMemFreeHost(hmem);
}

static void usage(const char *cmdname)
{
	fprintf(stderr,
			"usage: %s [<options> ..]\n"
			"\n"
			"options:\n"
			"  -d <device id>             (default: 0)\n"
			"  -m (sync|async)            (default: sync)\n"
			"  -n <number of trials>      (default: 100)\n"
			"  -s <size of buffer in MB>  (default: 128 = 128MB)\n"
			"  -c <size of chunks in KB>  (default: buffer size)\n",
			cmdname);
	exit(1);
}

int main(int argc, char *argv[])
{
	int				device_id = 0;
	CUdevice		device;
	CUcontext		context = NULL;
	CUstream		stream = NULL;
	CUresult		rc;
	int				c;
	char			namebuf[1024];

	while ((c = getopt(argc, argv, "d:m:n:s:c:")) >= 0)
	{
		switch (c)
		{
			case 'd':
				device_id = atoi(optarg);
				break;
			case 'm':
				if (strcmp(optarg, "sync") == 0)
					is_blocking = 1;
				else if (strcmp(optarg, "async") == 0)
					is_blocking = 0;
				else
					usage(basename(argv[0]));
				break;
			case 'n':
				num_trial = atoi(optarg);
				break;
			case 's':
				buffer_size = atoi(optarg) << 20;
				break;
			case 'c':
				chunk_size = atoi(optarg) << 10;
				break;
			default:
				usage(basename(argv[0]));
				break;
		}
	}
	if (optind != argc)
		usage(basename(argv[0]));

	if (chunk_size == 0)
		chunk_size = buffer_size;
	else if (buffer_size % chunk_size != 0 || buffer_size < chunk_size)
	{
		fprintf(stderr, "chunk_size (-c) must be aligned to buffer_size\n");
		return 1;
	}

	/*
	 * Initialize CUDA device
	 */
	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		error_exit("failed on cuInit : %s", cuGetErrorString(rc));

	rc = cuDeviceGet(&device, device_id);
	if (rc != CUDA_SUCCESS)
		error_exit("failed on cuDeviceGet(%d) : %s",
				   device_id, cuGetErrorString(rc));

	/* Get name of cuda device */
	rc = cuDeviceGetName(namebuf, sizeof(namebuf), device);
	if (rc != CUDA_SUCCESS)
		error_exit("failed on cuDeviceGetName : %s", cuGetErrorString(rc));

	/* Construct an CUDA context */
	rc = cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device);
	if (rc != CUDA_SUCCESS)
		error_exit("failed on cuCtxCreate : %s", cuGetErrorString(rc));

	rc = cuCtxSetCurrent(context);
	if (rc != CUDA_SUCCESS)
		error_exit("failed on cuCtxSetCurrent : %s", cuGetErrorString(rc));

	/* do the job */
	run_test(namebuf, context, stream);

	return 0;
}
