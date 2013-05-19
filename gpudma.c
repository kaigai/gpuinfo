/*
 * gpudma - test for DMA transfer
 */
#include <errno.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <CL/cl.h>

#define lengthof(array) (sizeof (array) / sizeof ((array)[0]))
#define error_exit(fmt,...)					\
	do {									\
		fprintf(stderr, "%s:%d " fmt "\n",	\
				__FUNCTION__, __LINE__,		\
				##__VA_ARGS__);				\
		exit(1);							\
	} while(0)

extern const char *opencl_strerror(cl_int errcode);

static cl_int	platform_idx = 1;
static cl_int	device_idx = 1;
static cl_bool	is_blocking = CL_TRUE;
static cl_int	num_trial = 100;			/* 100 times */
static size_t	buffer_size = 128 << 20;	/* 128MB */

static void
run_test(const char *namebuf, cl_context context, cl_command_queue cmdq)
{
	cl_event	   *ev;
	char		   *hmem;
	cl_mem			dmem;
	cl_mem			pinned = NULL;
	cl_int			rc, i, j;
	struct timeval	tv1, tv2;

	ev = malloc(sizeof(cl_event) * 2 * num_trial);
	if (!ev)
		error_exit("out of memory (%s)", strerror(rc));

	hmem = malloc(buffer_size);
	if (!hmem)
		error_exit("out of memory (%s)", strerror(rc));

	dmem = clCreateBuffer(context,
						  CL_MEM_READ_WRITE,
						  buffer_size,
						  NULL,
						  &rc);
	if (rc != CL_SUCCESS)
		error_exit("failed on clCreateBuffer(size=%lu) (%s)",
				   buffer_size, opencl_strerror(rc));

	gettimeofday(&tv1, NULL);

	if (!is_blocking)
	{
		pinned = clCreateBuffer(context,
								CL_MEM_READ_WRITE |
								CL_MEM_USE_HOST_PTR,
								buffer_size,
								hmem,
								&rc);
		if (rc != CL_SUCCESS)
			error_exit("failed on clCreateBuffer(size=%lu) (%s)",
					   buffer_size, opencl_strerror(rc));
	}

	for (i=0, j=0; i < num_trial; i++)
	{
		rc = clEnqueueWriteBuffer(cmdq,
								  dmem,
								  is_blocking,
								  0,
								  buffer_size,
								  hmem,
								  j > 0 ? 1 : 0,
								  j > 0 ? ev + j - 1 : NULL,
								  ev + j);
		if (rc != CL_SUCCESS)
			error_exit("failed on clEnqueueWriteBuffer (%s)",
					   opencl_strerror(rc));
		j++;

		rc = clEnqueueReadBuffer(cmdq,
								 dmem,
								 is_blocking,
								 0,
								 buffer_size,
								 hmem,
								 j > 0 ? 1 : 0,
								 j > 0 ? ev + j - 1 : NULL,
								 ev + j);
		if (rc != CL_SUCCESS)
			error_exit("failed on clEnqueueReadBuffer (%s)",
					   opencl_strerror(rc));
		j++;
	}
	rc = clFinish(cmdq);
	if (rc != CL_SUCCESS)
		error_exit("failed on clFinish (%s)", opencl_strerror(rc));

	gettimeofday(&tv2, NULL);

	printf("DMA send/recv test result\n"
		   "device:         %s\n"
		   "size:           %luMB\n"
		   "ntrials:        %d\n"
		   "total_size:     %luMB\n"
		   "time:           %.2fs\n"
		   "speed:          %.2fMB/s\n"
		   "mode:           %s\n",
		   namebuf,
		   buffer_size >> 20,
		   num_trial,
		   (buffer_size >> 20) * num_trial,
		   (double)((tv2.tv_sec * 1000000 + tv2.tv_usec) -
					(tv1.tv_sec * 1000000 + tv1.tv_usec)) / 1000000.0,
		   (double)(((buffer_size >> 20) * num_trial) * 1000000) /
           (double)((tv2.tv_sec * 1000000 + tv2.tv_usec) -
                    (tv1.tv_sec * 1000000 + tv1.tv_usec)),
		   is_blocking ? "sync" : "async");

	/* release resources */
	clReleaseMemObject(dmem);
	free(hmem);
	free(ev);
}

static void usage(const char *cmdname)
{
	fprintf(stderr,
			"usage: %s [<options> ..]\n"
			"\n"
			"options:\n"
			"  -p <platform index>    (default: 1)\n"
			"  -d <device index>      (default: 1)\n"
			"  -m (sync|async)        (default: sync)\n"
			"  -n <number of trials>  (default: 100)\n"
			"  -s <size of buffer>    (default: 128 = 128MB)\n",
			cmdname);
	exit(1);
}

int main(int argc, char *argv[])
{
	cl_platform_id	platform_ids[32];
	cl_int			platform_num;
	cl_device_id	device_ids[256];
	cl_int			device_num;
	cl_context		context;
	cl_command_queue cmdq;
	cl_int			c, rc;
	char			namebuf[1024];

	while ((c = getopt(argc, argv, "p:d:m:n:s:")) >= 0)
	{
		switch (c)
		{
			case 'p':
				platform_idx = atoi(optarg);
				break;
			case 'd':
				device_idx = atoi(optarg);
				break;
			case 'm':
				if (strcmp(optarg, "sync") == 0)
					is_blocking = CL_TRUE;
				else if (strcmp(optarg, "async") == 0)
					is_blocking = CL_FALSE;
				else
					usage(basename(argv[0]));
				break;
			case 'n':
				num_trial = atoi(optarg);
				break;
			case 's':
				buffer_size = atoi(optarg) << 20;
				break;
			default:
				usage(basename(argv[0]));
				break;
		}
	}
	if (optind != argc)
		usage(basename(argv[0]));

	/*
	 * Initialize OpenCL platform/device
	 */
	opencl_entry_init();

	/* Get platform IDs */
	rc = clGetPlatformIDs(lengthof(platform_ids),
						  platform_ids,
						  &platform_num);
	if (rc != CL_SUCCESS)
		error_exit("failed on clGetPlatformIDs (%s)", opencl_strerror(rc));
	if (platform_idx < 1 || platform_idx > platform_num)
		error_exit("opencl platform index %d did not exist", platform_idx);

	/* Get device IDs */
	rc = clGetDeviceIDs(platform_ids[platform_idx - 1],
						CL_DEVICE_TYPE_ALL,
						lengthof(device_ids),
						device_ids,
						&device_num);
	if (rc != CL_SUCCESS)
		error_exit("failed on clGetDeviceIDs (%s)\n", opencl_strerror(rc));
	if (device_idx < 1 || device_idx > device_num)
		error_exit("opencl device index %d did not exist", device_idx);

	/* Get name of opencl device */
	rc = clGetDeviceInfo(device_ids[device_idx - 1],
						 CL_DEVICE_NAME,
						 sizeof(namebuf), namebuf, NULL);
	if (rc != CL_SUCCESS)
		error_exit("failed on clGetDeviceInfo (%s)", opencl_strerror(rc));

	/* Construct an OpenCL context */
	context = clCreateContext(NULL,
                              1,
                              &device_ids[device_idx - 1],
                              NULL,
                              NULL,
                              &rc);
	if (rc != CL_SUCCESS)
		error_exit("failed to create an opencl context (%s)",
				   opencl_strerror(rc));

	/* Construct an OpenCL command queue */
	cmdq = clCreateCommandQueue(context,
								device_ids[device_idx - 1],
								CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
								&rc);
	if (rc != CL_SUCCESS)
		error_exit("failed to create an opencl command queue (%s)",
				   opencl_strerror(rc));

	/* do the job */
	run_test(namebuf, context, cmdq);

	/* cleanup resources */
	clReleaseCommandQueue(cmdq);
	clReleaseContext(context);

	return 0;
}
