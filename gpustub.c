#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>

#define lengthof(array) (sizeof (array) / sizeof ((array)[0]))

extern const char *opencl_strerror(cl_int errcode);

static const char *kernel_source =
	"__kernel void\n"
	"kernel_test(__global uint *arg)\n"
	"{\n"
	"  arg[get_global_id(0)] = (get_global_size(0) -\n"
	"                           get_global_id(0));\n"
	"}\n";

#define MEM_SIZE	2048

typedef struct {
	cl_kernel	kernel;
	cl_mem		dmem;
	cl_event	ev[4];
	cl_uint		hmem[MEM_SIZE];
} clstate_t;

static void
cb_kernel_complete(cl_event event, cl_int status, void *user_data)
{
	clstate_t  *clstate = user_data;
	int			i;

	for (i=0; i < MEM_SIZE; i++)
		printf(" %u", clstate->hmem[i]);
	putchar('\n');

	clReleaseKernel(clstate->kernel);
	clReleaseMemObject(clstate->dmem);
	clReleaseEvent(clstate->ev[0]);
	clReleaseEvent(clstate->ev[1]);
	clReleaseEvent(clstate->ev[2]);
	free(clstate);
}

static int
run_opencl_kernel(cl_context context, cl_device_id device)
{
	cl_command_queue cmdq;
	cl_program	program;
	clstate_t  *clstate;
	size_t		gwork_sz = MEM_SIZE;
	size_t		lwork_sz = MEM_SIZE / 4;
	size_t		source_len = strlen(kernel_source);
	cl_int		rc;

	cmdq = clCreateCommandQueue(context, device,
								CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
								CL_QUEUE_PROFILING_ENABLE,
								&rc);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clCreateCommandQueue (%s)\n",
				opencl_strerror(rc));
		return 1;
	}

	program = clCreateProgramWithSource(context,
										1,
										&kernel_source,
										&source_len,
										&rc);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clCreateProgramWithSource (%s)\n",
				opencl_strerror(rc));
		return 1;
	}

	rc = clBuildProgram(program,
						1,
						&device,
						NULL,
						NULL,
						NULL);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clBuildProgram(%s)\n",
				opencl_strerror(rc));
		if (rc == CL_BUILD_PROGRAM_FAILURE )
		{
			char	buffer[65536];

			rc = clGetProgramBuildInfo(program,
									   device,
									   CL_PROGRAM_BUILD_LOG,
									   sizeof(buffer),
									   buffer,
									   NULL);
			if (rc == CL_SUCCESS)
				fputs(buffer, stderr);
		}
		return 1;
	}

retry:
	clstate = malloc(sizeof(clstate_t));
	if (!clstate)
	{
		fprintf(stderr, "out of memory");
		return 1;
	}

	clstate->kernel = clCreateKernel(program,
									 "kernel_test",
									 &rc);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clCreateKernel (%s)",
				opencl_strerror(rc));
		return 1;
	}

	clstate->dmem = clCreateBuffer(context,
								   CL_MEM_READ_WRITE,
								   sizeof(cl_uint) * MEM_SIZE,
								   NULL,
								   &rc);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clCreateBuffer (%s)",
				opencl_strerror(rc));
		return 1;
	}

	rc = clSetKernelArg(clstate->kernel,
						0,
						sizeof(cl_mem),
						&clstate->dmem);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clSetKernelArg (%s)",
				opencl_strerror(rc));
		return 1;
	}

	/* OK, enqueue kernel */
	rc = clEnqueueWriteBuffer(cmdq,
							  clstate->dmem,
							  CL_FALSE,
							  0,
							  sizeof(cl_uint) * MEM_SIZE,
							  clstate->hmem,
							  0,
							  NULL,
							  &clstate->ev[0]);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clEnqueueWriteBuffer (%s)",
				opencl_strerror(rc));
		return 1;
	}

	rc = clEnqueueNDRangeKernel(cmdq,
								clstate->kernel,
								1,
								NULL,
								&gwork_sz,
								&lwork_sz,
								1,
								&clstate->ev[0],
								&clstate->ev[1]);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clEnqueueNDRangeKernel (%s)",
				opencl_strerror(rc));
		return 1;
	}

	rc = clEnqueueReadBuffer(cmdq,
							 clstate->dmem,
							 CL_FALSE,
							 0,
							 sizeof(cl_uint) * MEM_SIZE,
							 clstate->hmem,
							 1,
							 &clstate->ev[1],
							 &clstate->ev[2]);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clEnqueueReadBuffer (%s)",
				opencl_strerror(rc));
		return 1;
	}

	rc = clSetEventCallback(clstate->ev[2],
							CL_COMPLETE,
							cb_kernel_complete,
							clstate);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clSetEventCallback (%s)",
				opencl_strerror(rc));
		return 1;
	}
	sleep(15);
	goto retry;
}

int main(int argc, char *argv[])
{
	cl_platform_id	platforms[32];
	cl_device_id	devices[32];
	cl_context		context;
	cl_int		num_platforms;
	cl_int		num_devices;
	cl_int		pindex = 0;
	cl_int		dindex = 0;
	cl_int		i, c, rc;

	while ((c = getopt(argc, argv, "p:d:")) != -1)
	{
		switch (c)
		{
			case 'p':
				pindex = atoi(optarg);
				break;
			case 'd':
				dindex = atoi(optarg);
				break;
			default:
				fprintf(stderr, "usage: %s [-p <platform>] [-d <device>]\n",
						basename(argv[0]));
				return 1;
		}
	}
	opencl_entry_init();

	rc = clGetPlatformIDs(lengthof(platforms),
						  platforms,
						  &num_platforms);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clGetPlatformIDs (%s)\n",
				opencl_strerror(rc));
		return 1;
	}
	if (pindex < 0 || pindex >= num_platforms)
	{
		fprintf(stderr, "platform (%d) is not valid\n", pindex);
		return 1;
	}

	rc = clGetDeviceIDs(platforms[pindex],
						CL_DEVICE_TYPE_ALL,
						lengthof(devices),
						devices,
						&num_devices);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clGetDeviceIDs (%s)\n",
				opencl_strerror(rc));
		return 1;
	}
	if (dindex < 0 || dindex >= num_devices)
	{
		fprintf(stderr, "device (%d) is not valid\n", dindex);
		return 1;
	}

	context = clCreateContext(NULL,
							  num_devices,
							  devices,
							  NULL,
							  NULL,
							  &rc);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed to create opencl context (%s)",
				opencl_strerror(rc));
		return 1;
	}
	return run_opencl_kernel(context, devices[dindex]);
}
