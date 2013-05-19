#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <CL/cl.h>

#define lengthof(array) (sizeof (array) / sizeof ((array)[0]))

extern const char *opencl_strerror(cl_int errcode);

static int		platform_idx = 1;
static int		device_idx = 1;
static char	   *cl_build_opts = "-Werror";

static int
opencl_compile(cl_context context,
			   cl_device_id device_id,
			   const char *filename)
{
	cl_program	program;
	int			fdesc;
	char	   *source;
	size_t		length;
	cl_int		rc;
	struct stat	stbuf;
	cl_build_status status;
	char		logbuf[65536];
	size_t		loglen;

	fdesc = open(filename, O_RDONLY);
	if (fdesc < 0)
	{
		fprintf(stderr, "failed to open '%s' (%s)\n",
				filename, strerror(errno));
		return 1;
	}

	if (fstat(fdesc, &stbuf) != 0)
	{
		fprintf(stderr, "failed to fstat on '%s' (%s)\n",
				filename, strerror(errno));
		return 1;
	}
	length = stbuf.st_size;

	source = malloc(length);
	if (!source)
	{
		fprintf(stderr, "out of memory (%s)\n",
				strerror(errno));
		return 1;
	}

	if (read(fdesc, source, length) != length)
	{
		fprintf(stderr, "failed to read whole of source file (%s)\n",
				strerror(errno));
		return 1;
	}

	/* make a program object */
	program = clCreateProgramWithSource(context,
										1,
										(const char **)&source,
										&length,
										&rc);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clCreateProgramWithSource (%s)\n",
				opencl_strerror(rc));
		return 1;
	}

	/* build this program */
	rc = clBuildProgram(program,
						1,
						&device_id,
						cl_build_opts,
						NULL,
						NULL);
	if (rc != CL_SUCCESS && rc != CL_BUILD_PROGRAM_FAILURE)
	{
		if (rc == CL_INVALID_BUILD_OPTIONS)
			fprintf(stderr,
					"failed on clBuildProgram with build options: %s (%s)",
					cl_build_opts, opencl_strerror(rc));
		else
			fprintf(stderr, "failed on clBuildProgram (%s)\n",
					opencl_strerror(rc));
		return 1;
	}

	/* Get status and logs */
	rc = clGetProgramBuildInfo(program,
							   device_id,
							   CL_PROGRAM_BUILD_STATUS,
							   sizeof(cl_build_status),
							   &status,
							   NULL);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clGetProgramBuildInfo (%s)\n",
				opencl_strerror(rc));
		return 1;
	}

	rc = clGetProgramBuildInfo(program,
							   device_id,
							   CL_PROGRAM_BUILD_LOG,
							   sizeof(logbuf),
							   &logbuf,
							   &loglen);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clGetProgramBuildInfo(%s)\n",
				opencl_strerror(rc));
		return 1;
	}

	switch (status)
	{
		case CL_BUILD_NONE:
			puts("build none");
			break;
		case CL_BUILD_ERROR:
			puts("build error");
			break;
		case CL_BUILD_SUCCESS:
			puts("build success");
			break;
		case CL_BUILD_IN_PROGRESS:
			puts("build in progress");
			break;
		default:
			puts("unknown");
			break;
	}
	write(fileno(stdout), logbuf, loglen);
	
	clReleaseProgram(program);
	free(source);
	return 0;
}

int main(int argc, char *argv[])
{
	cl_platform_id	platform_ids[32];
	cl_int			platform_num;
	cl_device_id	device_ids[256];
	cl_int			device_num;
	cl_context		context;
	cl_int			code, rc, i;
	char			namebuf[1024];

	while ((code = getopt(argc, argv, "p:d:o:")) >= 0)
	{
		switch (code)
		{
			case 'p':
				platform_idx = atoi(optarg);
				break;
			case 'd':
				device_idx = atoi(optarg);
				break;
			case 'o':
				cl_build_opts = optarg;
				break;
			default:
				fprintf(stderr,
						"usage: %s "
						"[-p <pf_idx>][-d <dev_idx>]"
						"[-o <build_opts] <source>\n",
						basename(argv[0]));
				return 1;
		}
	}
	if (optind >= argc) {
		fprintf(stderr, "no source files were given.\n");
		return 1;
	}
	opencl_entry_init();

	/* Get platform IDs */
	rc = clGetPlatformIDs(lengthof(platform_ids),
						  platform_ids,
						  &platform_num);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clGetPlatformIDs (%s)",
				opencl_strerror(rc));
		return 1;
	}
	if (platform_idx < 1 || platform_idx > platform_num)
	{
		fprintf(stderr, "opencl platform index %d did not exist.\n");
		return 1;
	}

	/* Get device IDs */
	rc = clGetDeviceIDs(platform_ids[platform_idx - 1],
						CL_DEVICE_TYPE_ALL,
						lengthof(device_ids),
						device_ids,
						&device_num);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clGetDeviceIDs (%s)\n",
				opencl_strerror(rc));
		return 1;
	}
	if (device_idx < 1 || device_idx > device_num)
	{
		fprintf(stderr, "opencl device index %d did not exist.\n");
		return 1;
	}

	/* Print name of opencl platform */
	rc = clGetPlatformInfo(platform_ids[platform_idx - 1],
						   CL_PLATFORM_NAME,
						   sizeof(namebuf), namebuf, NULL);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clGetPlatformInfo (%s)\n",
				opencl_strerror(rc));
		return 1;
	}
	printf("platform: %s\n", namebuf);

	/* Print name of opencl device */
	rc = clGetDeviceInfo(device_ids[device_idx - 1],
						 CL_DEVICE_NAME,
						 sizeof(namebuf), namebuf, NULL);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clGetDeviceInfo (%s)\n",
				opencl_strerror(rc));
		return 1;
	}
	printf("device: %s\n", namebuf);

	/* create an opencl context */
	context = clCreateContext(NULL,
							  1,
							  device_ids + (device_idx - 1),
							  NULL,
							  NULL,
							  &rc);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed to create an opencl context (%s)\n",
				opencl_strerror(rc));
		return 1;
	}

	/* do the jobs */
	for (i = optind; i < argc; i++)
	{
		printf("source: %s ... ", argv[i]);
		if (opencl_compile(context, device_ids[device_idx - 1], argv[i]) != 0)
		{
			puts("error");
			return 1;
		}
		putchar('\n');
	}
	clReleaseContext(context);
}
