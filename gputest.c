/*
 * gputest.c - test module for OpenCL/CUDA functionalities
 */
//#define GPUTEST_CUDA	1
#define GPUTEST_OPENCL	1

#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/ipc.h"
#include <sys/time.h>
#ifdef GPUTEST_CUDA
#include <cuda.h>
#endif
#ifdef GPUTEST_OPENCL
#include <CL/cl.h>
#endif

PG_MODULE_MAGIC;

/* declarations */
extern Datum gputest_init_opencl(PG_FUNCTION_ARGS);
extern Datum gputest_dmasend_opencl(PG_FUNCTION_ARGS);
extern Datum gputest_cleanup_opencl(PG_FUNCTION_ARGS);
extern void  _PG_init(void);

static shmem_startup_hook_type shmem_startup_hook_next;

#define TIMEVAL_DIFF(tv2,tv1)											\
	(((double)((tv2)->tv_sec * 1000000L + (tv2)->tv_usec) -				\
	  (double)((tv1)->tv_sec * 1000000L + (tv1)->tv_usec)) / 1000000.0)

#ifdef GPUTEST_CUDA
static bool			cuda_initialized = false;
static CUdevice		cuda_device;
static CUcontext	cuda_context = NULL;

/* why CUDA 6.5 lacks declaration? */
extern CUresult cuGetErrorString(CUresult error, const char** pStr);

static const char *
cuda_strerror(CUresult rc)
{
	static char	buffer[256];
	const char *result;

	if (cuGetErrorString(rc, &result) != CUDA_SUCCESS)
	{
		snprintf(buffer, sizeof(buffer), "cuda error (%d)", rc);
		return buffer;
	}
	return result;
}
#endif
#ifdef GPUTEST_OPENCL
static cl_platform_id	opencl_platform_id;
static cl_device_id		opencl_device_id;
static cl_context		opencl_context = NULL;
#endif

Datum
gputest_init_opencl(PG_FUNCTION_ARGS)
{
#ifdef GPUTEST_CUDA
	CUresult	rc;
	struct timeval tv1, tv2;

	if (!cuda_initialized)
	{
		rc = cuInit(0);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuInit: %s", cuda_strerror(rc));
		cuda_initialized = true;
	}

	if (!cuda_context)
	{
		rc = cuDeviceGet(&cuda_device, 0);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuDeviceGet: %s", cuda_strerror(rc));

		rc = cuCtxCreate(&cuda_context, 0, cuda_device);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxCreate: %s", cuda_strerror(rc));

		rc = cuCtxSetCurrent(cuda_context);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxSetCurrent: %s", cuda_strerror(rc));

		gettimeofday(&tv1, NULL);
		rc = cuMemHostRegister(BufferBlocks, NBuffers * (Size) BLCKSZ,
							   CU_MEMHOSTREGISTER_PORTABLE);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "cuMemHostRegister: %s", cuda_strerror(rc));
		gettimeofday(&tv2, NULL);
		elog(INFO, "cuMemHostRegister takes %.2fsec to map %zuGB",
			 TIMEVAL_DIFF(&tv2, &tv1), ((Size)NBuffers * (Size) BLCKSZ) >> 30);
	}
	else
		elog(INFO, "we already have cuda context");
#endif
#ifdef GPUTEST_OPENCL
	cl_int			rc;
	struct timeval	tv1, tv2;

	if (!opencl_context)
	{
		rc = clGetPlatformIDs(1, &opencl_platform_id, NULL);
		if (rc != CL_SUCCESS)
			elog(ERROR, "failed on clGetPlatformIDs: %d", rc);

		rc = clGetDeviceIDs(opencl_platform_id,
							CL_DEVICE_TYPE_ALL,
							1,
							&opencl_device_id,
							NULL);
		if (rc != CL_SUCCESS)
			elog(ERROR, "failed on clGetDeviceIDs: %d", rc);

		opencl_context = clCreateContext(NULL,
										 1,
										 &opencl_device_id,
										 NULL,
										 NULL,
										 &rc);
		if (rc != CL_SUCCESS)
			elog(ERROR, "failed on clCreateContext: %d", rc);
	}
	gettimeofday(&tv1, NULL);
	(void) clCreateBuffer(opencl_context,
						  CL_MEM_READ_WRITE |
						  CL_MEM_USE_HOST_PTR,
						  NBuffers * (Size) BLCKSZ,
						  BufferBlocks,
						  &rc);
	if (rc != CL_SUCCESS)
		elog(ERROR, "failed on clCreateBuffer: %d", rc);
	gettimeofday(&tv2, NULL);
	elog(LOG, "clCreateBuffer takes %.2fsec to map %zuGB",
		 TIMEVAL_DIFF(&tv2, &tv1), ((Size)NBuffers * (Size) BLCKSZ) >> 30);
#endif
	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(gputest_init_opencl);

Datum
gputest_dmasend_opencl(PG_FUNCTION_ARGS)
{
#ifdef GPUTEST_CUDA
	CUdevice	device;
	CUcontext	context;
	CUstream	stream;
	CUdeviceptr	daddr;
	CUevent		start;
	CUevent		stop;
	CUresult	rc;
	int			loop;
	float		elapsed;
	Size		unitsz = 100 * 1024 * 1024; //100MB
	Size		offset;

	if (!cuda_initialized)
	{
		rc = cuInit(0);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuInit: %s", cuda_strerror(rc));
		cuda_initialized = true;
	}

	rc = cuDeviceGet(&device, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGet: %s", cuda_strerror(rc));

	rc = cuCtxCreate(&context, 0, device);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxCreate: %s", cuda_strerror(rc));

	rc = cuCtxSetCurrent(context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxSetCurrent: %s", cuda_strerror(rc));

	rc = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
	if (rc != CUDA_SUCCESS)
        elog(ERROR, "failed on cuStreamCreate: %s", cuda_strerror(rc));

	rc = cuMemAlloc(&daddr, unitsz);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemAlloc: %s", cuda_strerror(rc));

	rc = cuEventCreate(&start, CU_EVENT_DEFAULT);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventCreate: %s", cuda_strerror(rc));

	rc = cuEventCreate(&stop, CU_EVENT_DEFAULT);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventCreate: %s", cuda_strerror(rc));

	rc = cuEventRecord(start, stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventRecord: %s", cuda_strerror(rc));

	for (loop = 0; loop < 1; loop++)
	{
		for (offset = 0;
			 offset < NBuffers * (Size) BLCKSZ - unitsz;
			 offset += unitsz)
		{
			rc = cuMemcpyHtoDAsync(daddr,
								   BufferBlocks + offset,
								   unitsz,
								   stream);
			if (rc != CUDA_SUCCESS)
				elog(ERROR, "failed on cuMemcpyHtoDAsync: %s",
					 cuda_strerror(rc));
		}
	}
	rc = cuEventRecord(stop, stream);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventRecord: %s", cuda_strerror(rc));

	rc = cuStreamSynchronize(stream);
	if (rc !=  CUDA_SUCCESS)
		elog(ERROR, "failed on cuStreamSynchronize: %s", cuda_strerror(rc));

	rc = cuEventElapsedTime (&elapsed, start, stop);
	if (rc !=  CUDA_SUCCESS)
		elog(ERROR, "failed on cuEventElapsedTime: %s", cuda_strerror(rc));
	elapsed /= 1000000.0;	/* sec */

	rc = cuMemFree(daddr);
	if (rc !=  CUDA_SUCCESS)
		elog(ERROR, "failed on cuMemFree: %s", cuda_strerror(rc));

	elog(INFO, "%zu GB DMA took %.2f sec (%.2f GB/sec)",
		 (loop * NBuffers * (Size) BLCKSZ) >> 30,
		 elapsed,
		 (double)((loop * NBuffers * (Size) BLCKSZ) >> 30) / elapsed);

	rc = cuCtxDestroy(context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxDestroy: %s", cuda_strerror(rc));
#endif
#ifdef GPUTEST_OPENCL
	cl_int			rc;
	struct timeval	tv1, tv2;

	if (!opencl_context)
	{
		rc = clGetPlatformIDs(1, &opencl_platform_id, NULL);
		if (rc != CL_SUCCESS)
			elog(ERROR, "failed on clGetPlatformIDs: %d", rc);

		rc = clGetDeviceIDs(opencl_platform_id,
							CL_DEVICE_TYPE_ALL,
							1,
							&opencl_device_id,
							NULL);
		if (rc != CL_SUCCESS)
			elog(ERROR, "failed on clGetDeviceIDs: %d", rc);

		opencl_context = clCreateContext(NULL,
										 1,
										 &opencl_device_id,
										 NULL,
										 NULL,
										 &rc);
		if (rc != CL_SUCCESS)
			elog(ERROR, "failed on clCreateContext: %d", rc);
	}
	gettimeofday(&tv1, NULL);
	(void) clCreateBuffer(opencl_context,
						  CL_MEM_READ_WRITE |
						  CL_MEM_USE_HOST_PTR,
						  NBuffers * (Size) BLCKSZ,
						  BufferBlocks,
						  &rc);
	if (rc != CL_SUCCESS)
		elog(ERROR, "failed on clCreateBuffer: %d", rc);
	gettimeofday(&tv2, NULL);
	elog(LOG, "clCreateBuffer takes %.2fsec to map %zuGB",
		 TIMEVAL_DIFF(&tv2, &tv1), ((Size)NBuffers * (Size) BLCKSZ) >> 30);
#endif
	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(gputest_dmasend_opencl);

Datum
gputest_cleanup_opencl(PG_FUNCTION_ARGS)
{
#ifdef GPUTEST_CUDA
	CUresult	rc;

	if (!cuda_initialized)
	{
		rc = cuInit(0);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuInit: %s", cuda_strerror(rc));
		cuda_initialized = true;
	}

	if (cuda_context)
	{
		rc = cuCtxDestroy(cuda_context);
		if (rc != CUDA_SUCCESS)
			elog(ERROR, "failed on cuCtxDestroy: %s", cuda_strerror(rc));
		cuda_context = NULL;
	}
	else
		elog(INFO, "no cuda context exists");
#endif
#ifdef GPUTEST_OPENCL



#endif
	PG_RETURN_NULL();
}
PG_FUNCTION_INFO_V1(gputest_cleanup_opencl);

static void
gputest_init(void)
{
#ifdef GPUTEST_CUDA
	int			n_devices;
	CUdevice	devices[10];
	CUcontext	context;
	CUresult	rc;
	struct timeval tv1, tv2;

	if (shmem_startup_hook_next)
		(*shmem_startup_hook_next)();

	elog(LOG, "Loading GPU Tests");

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuInit: %s", cuda_strerror(rc));

	rc = cuDeviceGetCount(&n_devices);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGetCount: %s", cuda_strerror(rc));
	if (n_devices < 1)
		elog(ERROR, "no cuda device found");

	rc = cuDeviceGet(devices, 0);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuDeviceGet: %s", cuda_strerror(rc));

	rc = cuCtxCreate(&context, 0, devices[0]);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxCreate: %s", cuda_strerror(rc));

	rc = cuCtxSetCurrent(context);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "failed on cuCtxSetCurrent: %s", cuda_strerror(rc));

	elog(LOG, "%p %zu", BufferBlocks, NBuffers * (Size) BLCKSZ);
	gettimeofday(&tv1, NULL);
	rc = cuMemHostRegister(BufferBlocks, NBuffers * (Size) BLCKSZ,
						   CU_MEMHOSTREGISTER_PORTABLE);
	if (rc != CUDA_SUCCESS)
		elog(ERROR, "cuMemHostRegister: %s", cuda_strerror(rc));
	gettimeofday(&tv2, NULL);
	elog(LOG, "cuMemHostRegister takes %.2fsec to map %zuGB",
		 TIMEVAL_DIFF(&tv2, &tv1), ((Size)NBuffers * (Size) BLCKSZ) >> 30);
#endif
#ifdef GPUTEST_OPENCL
	cl_platform_id	platform_id;
	cl_device_id	device_id;
	cl_context		context;
	cl_int			rc;
	struct timeval	tv1, tv2;

	rc = clGetPlatformIDs(1, &platform_id, NULL);
	if (rc != CL_SUCCESS)
		elog(ERROR, "failed on clGetPlatformIDs: %d", rc);

	rc = clGetDeviceIDs(platform_id,
						CL_DEVICE_TYPE_ALL,
						1,
						&device_id,
						NULL);
	if (rc != CL_SUCCESS)
		elog(ERROR, "failed on clGetDeviceIDs: %d", rc);

	context = clCreateContext(NULL,
							  1,
							  &device_id,
							  NULL,
							  NULL,
							  &rc);
	if (rc != CL_SUCCESS)
		elog(ERROR, "failed on clCreateContext: %d", rc);

	gettimeofday(&tv1, NULL);
	(void) clCreateBuffer(context,
						  CL_MEM_READ_WRITE |
						  CL_MEM_USE_HOST_PTR,
						  NBuffers * (Size) BLCKSZ,
						  BufferBlocks,
						  &rc);
	if (rc != CL_SUCCESS)
		elog(ERROR, "failed on clCreateBuffer: %d", rc);
	gettimeofday(&tv2, NULL);
	elog(LOG, "clCreateBuffer takes %.2fsec to map %zuGB",
		 TIMEVAL_DIFF(&tv2, &tv1), ((Size)NBuffers * (Size) BLCKSZ) >> 30);
#endif
}

void
_PG_init(void)
{
	if (!process_shared_preload_libraries_in_progress)
		elog(ERROR, "gputest must be loaded via shared_preload_libraries");

	shmem_startup_hook_next = shmem_startup_hook;
    shmem_startup_hook = gputest_init;	
}
