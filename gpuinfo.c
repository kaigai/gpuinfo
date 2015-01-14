#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>

#define lengthof(array) (sizeof (array) / sizeof ((array)[0]))

extern const char *opencl_strerror(cl_int errcode);

static int only_list = 0;
static int only_platform = -1;
static int only_device = -1;
static struct {
	char	profile[256];
	char	version[256];
	char	name[256];
	char	vendor[256];
	char	extensions[1024];
} platform_info;
#define PLATFORM_ATTR(param,field)				\
	{ param, sizeof(platform_info.field), &(platform_info.field) }

static struct {
	cl_uint		address_bits;
	cl_bool		available;
	cl_bool		compiler_available;
	cl_device_fp_config	double_fp_config;
	cl_bool		endian_little;
	cl_bool		error_correction_support;
	cl_device_exec_capabilities execution_capabilities;
	char		extensions[1024];
	cl_ulong	global_mem_cache_size;
	cl_device_mem_cache_type global_mem_cache_type;
	cl_uint		global_mem_cacheline_size;
	cl_ulong	global_mem_size;
	cl_device_fp_config half_fp_config;
	cl_bool		host_unified_memory;
	cl_bool		image_support;
	size_t		image2d_max_height;
	size_t		image2d_max_width;
	size_t		image3d_max_depth;
	size_t		image3d_max_height;
	size_t		image3d_max_width;
	cl_ulong	local_mem_size;
	cl_device_local_mem_type local_mem_type;
	cl_uint		max_clock_frequency;
	cl_uint		max_compute_units;
	cl_uint		max_constant_args;
	cl_ulong	max_constant_buffer_size;
	cl_ulong	max_mem_alloc_size;
	size_t		max_parameter_size;
	cl_uint		max_read_image_args;
	cl_uint		max_samplers;
	size_t		max_work_group_size;
	cl_uint		max_work_item_dimensions;
	size_t		max_work_item_sizes[10];
	cl_uint		max_write_image_args;
	cl_uint		mem_base_addr_align;
	cl_uint		min_data_type_align_size;
	char		name[256];
	cl_uint		native_vector_width_char;
	cl_uint		native_vector_width_short;
	cl_uint		native_vector_width_int;
	cl_uint		native_vector_width_long;
	cl_uint		native_vector_width_float;
	cl_uint		native_vector_width_double;
	cl_uint		native_vector_width_half;
	char		opencl_c_version[256];
	cl_uint		preferred_vector_width_char;
	cl_uint		preferred_vector_width_short;
	cl_uint		preferred_vector_width_int;
	cl_uint		preferred_vector_width_long;
	cl_uint		preferred_vector_width_float;
	cl_uint		preferred_vector_width_double;
	cl_uint		preferred_vector_width_half;
	char		profile[256];
	size_t		profiling_timer_resolution;
	cl_command_queue_properties queue_properties;
	cl_device_fp_config single_fp_config;
	cl_device_type type;
	char		vendor[256];
	cl_uint		vendor_id;
	char		version[256];
	char		driver_version[256];
} dinfo;
#define DEVICE_ATTR(param,field)				\
	{ param, sizeof(dinfo.field), &(dinfo.field) }

static const char *dev_fp_config_str(cl_device_fp_config conf)
{
	static char	buf[256];
	size_t		offset = 0;

	buf[offset] = '\0';
	if (conf & CL_FP_DENORM)
		offset += sprintf(buf + offset,
						  "%sDenorm", offset > 0 ? "," : "");
	if (conf & CL_FP_INF_NAN)
		offset += sprintf(buf + offset, "%sINF/NaN",
						  offset > 0 ? ", " : "");
	if (conf & CL_FP_ROUND_TO_NEAREST)
		offset += sprintf(buf + offset, "%sR/nearest",
						  offset > 0 ? ", " : "");
	if (conf & CL_FP_ROUND_TO_ZERO)
		offset += sprintf(buf + offset, "%sR/zero",
						  offset > 0 ? ", " : "");
	if (conf & CL_FP_ROUND_TO_INF)
		offset += sprintf(buf + offset, "%sR/INF",
						  offset > 0 ? ", " : "");
	if (conf & CL_FP_FMA)
		offset += sprintf(buf + offset, "%sFMA",
						  offset > 0 ? ", " : "");
	return buf;
}

static const char *
dev_execution_capabilities_str(cl_device_exec_capabilities caps)
{
	return ((caps & CL_EXEC_KERNEL) != 0
			? ((caps & CL_EXEC_NATIVE_KERNEL) != 0
			   ? "kernel, native kernel"
			   : "kernel")
			: ((caps & CL_EXEC_NATIVE_KERNEL) != 0
			   ? "native kernel"
			   : "none"));
}

static const char *
dev_mem_cache_type_str(cl_device_mem_cache_type cache_type)
{
	switch (cache_type)
	{
		case CL_NONE:
			return "none";
		case CL_READ_ONLY_CACHE:
			return "read-only";
		case CL_READ_WRITE_CACHE:
			return "read-write";
		default:
			return "unknown";
	}
}

static const char *
dev_local_mem_type_str(cl_device_local_mem_type local_type)
{
	switch (local_type)
	{
		case CL_LOCAL:
			return "SRAM";
		case CL_GLOBAL:
			return "DRAM";
		default:
			return "unknown";
	}
}

static const char *
dev_type_str(cl_device_type dev_type)
{
	switch (dev_type)
	{
		case CL_DEVICE_TYPE_CPU:
			return "CPU";
		case CL_DEVICE_TYPE_GPU:
			return "GPU";
		case CL_DEVICE_TYPE_ACCELERATOR:
			return "Accelerator";
		case CL_DEVICE_TYPE_DEFAULT:
			return "Default";
		default:
			return "unknown";
	}
}

static const char *
dev_queue_properties_str(cl_command_queue_properties cmdq)
{
	static char	buf[256];
	size_t		offset = 0;

	buf[offset] = '\0';
	if (cmdq & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
		offset += sprintf(buf + offset, "%sout-of-order execution",
						  (offset > 0 ? ", " : ""));
	if (cmdq & CL_QUEUE_PROFILING_ENABLE)
		offset += sprintf(buf + offset, "%sprofiling",
						  (offset > 0 ? ", " : ""));
	return buf;
}

static void dump_device(int index, cl_device_id device_id)
{
	static struct {
		cl_device_info info;
		size_t		size;
		void	   *addr;
	} catalog[] = {
		DEVICE_ATTR(CL_DEVICE_ADDRESS_BITS, address_bits),
		DEVICE_ATTR(CL_DEVICE_AVAILABLE, available),
		DEVICE_ATTR(CL_DEVICE_COMPILER_AVAILABLE, compiler_available),
		DEVICE_ATTR(CL_DEVICE_DOUBLE_FP_CONFIG, double_fp_config),
		DEVICE_ATTR(CL_DEVICE_ENDIAN_LITTLE, endian_little),
		DEVICE_ATTR(CL_DEVICE_ERROR_CORRECTION_SUPPORT,
					error_correction_support),
		DEVICE_ATTR(CL_DEVICE_EXECUTION_CAPABILITIES,
					execution_capabilities),
		DEVICE_ATTR(CL_DEVICE_EXTENSIONS, extensions),
		DEVICE_ATTR(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, global_mem_cache_size),
		DEVICE_ATTR(CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, global_mem_cache_type),
		DEVICE_ATTR(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
					global_mem_cacheline_size),
		DEVICE_ATTR(CL_DEVICE_GLOBAL_MEM_SIZE, global_mem_size),
		DEVICE_ATTR(CL_DEVICE_HALF_FP_CONFIG, half_fp_config),
		DEVICE_ATTR(CL_DEVICE_HOST_UNIFIED_MEMORY, host_unified_memory),
		DEVICE_ATTR(CL_DEVICE_IMAGE_SUPPORT, image_support),
		DEVICE_ATTR(CL_DEVICE_IMAGE2D_MAX_HEIGHT, image2d_max_height),
		DEVICE_ATTR(CL_DEVICE_IMAGE2D_MAX_WIDTH, image2d_max_width),
		DEVICE_ATTR(CL_DEVICE_IMAGE3D_MAX_DEPTH, image3d_max_depth),
		DEVICE_ATTR(CL_DEVICE_IMAGE3D_MAX_HEIGHT, image3d_max_height),
		DEVICE_ATTR(CL_DEVICE_IMAGE3D_MAX_WIDTH, image3d_max_width),
		DEVICE_ATTR(CL_DEVICE_LOCAL_MEM_SIZE, local_mem_size),
		DEVICE_ATTR(CL_DEVICE_LOCAL_MEM_TYPE, local_mem_type),
		DEVICE_ATTR(CL_DEVICE_MAX_CLOCK_FREQUENCY, max_clock_frequency),
		DEVICE_ATTR(CL_DEVICE_MAX_COMPUTE_UNITS, max_compute_units),
		DEVICE_ATTR(CL_DEVICE_MAX_CONSTANT_ARGS, max_constant_args),
		DEVICE_ATTR(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
					max_constant_buffer_size),
		DEVICE_ATTR(CL_DEVICE_MAX_MEM_ALLOC_SIZE, max_mem_alloc_size),
		DEVICE_ATTR(CL_DEVICE_MAX_PARAMETER_SIZE, max_parameter_size),
		DEVICE_ATTR(CL_DEVICE_MAX_READ_IMAGE_ARGS, max_read_image_args),
		DEVICE_ATTR(CL_DEVICE_MAX_SAMPLERS, max_samplers),
		DEVICE_ATTR(CL_DEVICE_MAX_WORK_GROUP_SIZE, max_work_group_size),
		DEVICE_ATTR(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
					max_work_item_dimensions),
		DEVICE_ATTR(CL_DEVICE_MAX_WORK_ITEM_SIZES, max_work_item_sizes),
		DEVICE_ATTR(CL_DEVICE_MAX_WRITE_IMAGE_ARGS, max_write_image_args),
		DEVICE_ATTR(CL_DEVICE_MEM_BASE_ADDR_ALIGN, mem_base_addr_align),
		DEVICE_ATTR(CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
					min_data_type_align_size),
		DEVICE_ATTR(CL_DEVICE_NAME, name),
		DEVICE_ATTR(CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR,
					native_vector_width_char),
		DEVICE_ATTR(CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT,
					native_vector_width_short),
		DEVICE_ATTR(CL_DEVICE_NATIVE_VECTOR_WIDTH_INT,
					native_vector_width_int),
		DEVICE_ATTR(CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG,
					native_vector_width_long),
		DEVICE_ATTR(CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT,
					native_vector_width_float),
		DEVICE_ATTR(CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE,
					native_vector_width_double),
		DEVICE_ATTR(CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF,
					native_vector_width_half),
		DEVICE_ATTR(CL_DEVICE_OPENCL_C_VERSION, opencl_c_version),
		DEVICE_ATTR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
					preferred_vector_width_char),
		DEVICE_ATTR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
					preferred_vector_width_short),
		DEVICE_ATTR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
					preferred_vector_width_int),
		DEVICE_ATTR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
					preferred_vector_width_long),
		DEVICE_ATTR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
					preferred_vector_width_float),
		DEVICE_ATTR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
					preferred_vector_width_double),
		DEVICE_ATTR(CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,
					preferred_vector_width_half),
		DEVICE_ATTR(CL_DEVICE_PROFILE, profile),
		DEVICE_ATTR(CL_DEVICE_PROFILING_TIMER_RESOLUTION,
					profiling_timer_resolution),
		DEVICE_ATTR(CL_DEVICE_QUEUE_PROPERTIES, queue_properties),
		DEVICE_ATTR(CL_DEVICE_SINGLE_FP_CONFIG, single_fp_config),
		DEVICE_ATTR(CL_DEVICE_TYPE, type),
		DEVICE_ATTR(CL_DEVICE_VENDOR, vendor),
		DEVICE_ATTR(CL_DEVICE_VENDOR_ID, vendor_id),
		DEVICE_ATTR(CL_DEVICE_VERSION, version),
		DEVICE_ATTR(CL_DRIVER_VERSION, driver_version),
	};
	cl_int		i, rc;

	for (i=0; i < lengthof(catalog); i++)
	{
		rc = clGetDeviceInfo(device_id,
							 catalog[i].info,
							 catalog[i].size,
							 catalog[i].addr,
							 NULL);
		if (rc != CL_SUCCESS &&
			!(rc == CL_INVALID_VALUE &&
			  (catalog[i].info == CL_DEVICE_DOUBLE_FP_CONFIG ||
			   catalog[i].info == CL_DEVICE_HALF_FP_CONFIG)))
		{
			fprintf(stderr, "failed on clGetDeviceInfo (%s)\n",
					opencl_strerror(rc));
			exit(1);
		}
	}

	if (only_list)
		printf("  Device-%02d: %s / %s - %s\n",
			   index + 1,
			   dinfo.vendor,
			   dinfo.name,
			   dinfo.version);
	else
	{
		printf("  Device-%02d\n", index + 1);
		printf("  Device type:                     %s\n",
			   dev_type_str(dinfo.type));
		printf("  Vendor:                          %s (id: %08x)\n",
			   dinfo.vendor, dinfo.vendor_id);
		printf("  Name:                            %s\n",
			   dinfo.name);
		printf("  Version:                         %s\n",
			   dinfo.version);
		printf("  Driver version:                  %s\n",
			   dinfo.driver_version);
		printf("  OpenCL C version:                %s\n",
			   dinfo.opencl_c_version);
		printf("  Profile:                         %s\n",
			   dinfo.profile);
		printf("  Device available:                %s\n",
			   dinfo.available ? "yes" : "no");
		printf("  Address bits:                    %u\n",
			   dinfo.address_bits);
		printf("  Compiler available:              %s\n",
			   dinfo.compiler_available ? "yes" : "no");
		if (strstr(dinfo.extensions, "cl_khr_fp64") != NULL)
			printf("  Double FP config:                %s\n",
				   dev_fp_config_str(dinfo.double_fp_config));
		printf("  Endian:                          %s\n",
			   dinfo.endian_little ? "little" : "big");
		printf("  Error correction support:        %s\n",
			   dinfo.error_correction_support ? "yes" : "no");
		printf("  Execution capability:            %s\n",
			   dev_execution_capabilities_str(dinfo.execution_capabilities));
		printf("  Extensions:                      %s\n",
			   dinfo.extensions);
		printf("  Global memory cache size:        %lu KB\n",
			   dinfo.global_mem_cache_size / 1024);
		printf("  Global memory cache type:        %s\n",
			   dev_mem_cache_type_str(dinfo.global_mem_cache_type));
		printf("  Global memory cacheline size:    %u\n",
			   dinfo.global_mem_cacheline_size);
		printf("  Global memory size:              %zu MB\n",
			   dinfo.global_mem_size / (1024 * 1024));
		if (strstr(dinfo.extensions, "cl_khr_fp16") != NULL)
			printf("  Half FP config:                  %s\n",
				   dev_fp_config_str(dinfo.half_fp_config));
		printf("  Host unified memory:             %s\n",
			   dinfo.host_unified_memory ? "yes" : "no");
		printf("  Image support:                   %s\n",
			   dinfo.image_support ? "yes" : "no");
		printf("  Image 2D max size:               %lu x %lu\n",
			   dinfo.image2d_max_width,
			   dinfo.image2d_max_height);
		printf("  Image 3D max size:               %lu x %lu x %lu\n",
			   dinfo.image3d_max_width,
			   dinfo.image3d_max_height,
			   dinfo.image3d_max_depth);
		printf("  Local memory size:               %lu\n",
			   dinfo.local_mem_size);
		printf("  Local memory type:               %s\n",
			   dev_local_mem_type_str(dinfo.local_mem_type));
		printf("  Max clock frequency:             %u\n",
			   dinfo.max_clock_frequency);
		printf("  Max compute units:               %u\n",
			   dinfo.max_compute_units);
		printf("  Max constant args:               %u\n",
			   dinfo.max_constant_args);
		printf("  Max constant buffer size:        %zu\n",
			   dinfo.max_constant_buffer_size);
		printf("  Max memory allocation size:      %zu MB\n",
			   dinfo.max_mem_alloc_size / (1024 * 1024));
		printf("  Max parameter size:              %zu\n",
			   (cl_ulong)dinfo.max_parameter_size);
		printf("  Max read image args:             %u\n",
			   dinfo.max_read_image_args);
		printf("  Max samplers:                    %u\n",
			   dinfo.max_samplers);
		printf("  Max work-group size:             %zu\n",
			   (cl_ulong)dinfo.max_work_group_size);
		printf("  Max work-item sizes:             {%u,%u,%u}\n",
			   (cl_uint) dinfo.max_work_item_sizes[0],
			   (cl_uint) dinfo.max_work_item_sizes[1],
			   (cl_uint) dinfo.max_work_item_sizes[2]);
		printf("  Max write image args:            %u\n",
			   dinfo.max_write_image_args);
		printf("  Memory base address align:       %u\n",
			   dinfo.mem_base_addr_align);
		printf("  Min data type align size:        %u\n",
			   dinfo.min_data_type_align_size);
		printf("  Native vector width - char:      %u\n",
			   dinfo.native_vector_width_char);
		printf("  Native vector width - short:     %u\n",
			   dinfo.native_vector_width_short);
		printf("  Native vector width - int:       %u\n",
			   dinfo.native_vector_width_int);
		printf("  Native vector width - long:      %u\n",
			   dinfo.native_vector_width_long);
		printf("  Native vector width - float:     %u\n",
			   dinfo.native_vector_width_float);
		if (strstr(dinfo.extensions, "cl_khr_fp64") != NULL)
			printf("  Native vector width - double:    %u\n",
				   dinfo.native_vector_width_double);
		if (strstr(dinfo.extensions, "cl_khr_fp16") != NULL)
			printf("  Native vector width - half:      %u\n",
				   dinfo.native_vector_width_half);
		printf("  Preferred vector width - char:   %u\n",
			   dinfo.preferred_vector_width_char);
		printf("  Preferred vector width - short:  %u\n",
			   dinfo.preferred_vector_width_short);
		printf("  Preferred vector width - int:    %u\n",
			   dinfo.preferred_vector_width_int);
		printf("  Preferred vector width - long:   %u\n",
			   dinfo.preferred_vector_width_long);
		printf("  Preferred vector width - float:  %u\n",
			   dinfo.preferred_vector_width_float);
		if (strstr(dinfo.extensions, "cl_khr_fp64") != NULL)
			printf("  Preferred vector width - double: %u\n",
				   dinfo.preferred_vector_width_double);
		if (strstr(dinfo.extensions, "cl_khr_fp16") != NULL)
			printf("  Preferred vector width - half:   %u\n",
				   dinfo.preferred_vector_width_half);
		printf("  Profiling timer resolution:      %lu\n",
			   dinfo.profiling_timer_resolution);
		printf("  Queue properties:                %s\n",
			   dev_queue_properties_str(dinfo.queue_properties));
		printf("  Sindle FP config:                %s\n",
			   dev_fp_config_str(dinfo.single_fp_config));

	}
}

static void dump_platform(int index, cl_platform_id platform_id)
{
	static struct {
		cl_platform_info info;
		size_t		size;
		void	   *addr;
	} catalog[] = {
		PLATFORM_ATTR(CL_PLATFORM_PROFILE, profile),
		PLATFORM_ATTR(CL_PLATFORM_VERSION, version),
        PLATFORM_ATTR(CL_PLATFORM_NAME, name),
        PLATFORM_ATTR(CL_PLATFORM_VENDOR, vendor),
        PLATFORM_ATTR(CL_PLATFORM_EXTENSIONS, extensions),
	};
	cl_device_id	device_ids[256];
	cl_uint			device_num;
	cl_int			i, rc;

	for (i=0; i < lengthof(catalog); i++)
	{
		rc = clGetPlatformInfo(platform_id,
							   catalog[i].info,
							   catalog[i].size,
							   catalog[i].addr,
							   NULL);
		if (rc != CL_SUCCESS)
		{
			fprintf(stderr, "failed on clGetPlatformInfo (%s)\n",
					opencl_strerror(rc));
			exit(1);
		}
	}

	rc = clGetDeviceIDs(platform_id,
						CL_DEVICE_TYPE_ALL,
						lengthof(device_ids),
						device_ids,
						&device_num);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clGetDeviceIDs (%s)\n",
				opencl_strerror(rc));
		exit(1);
	}

	if (only_list)
		printf("Platform-%02d: %s / %s - %s\n", index + 1,
			   platform_info.vendor,
			   platform_info.name,
			   platform_info.version);
	else
	{
		printf("platform-index:      %d\n", index + 1);
		printf("platform-vendor:     %s\n", platform_info.vendor);
		printf("platform-name:       %s\n", platform_info.name);
		printf("platform-version:    %s\n", platform_info.version);
		printf("platform-profile:    %s\n", platform_info.profile);
		printf("platform-extensions: %s\n", platform_info.extensions);
	}

	for (i=0; i < device_num; i++)
	{
		if (only_device < 0 || i + 1 == only_device)
			dump_device(i, device_ids[i]);
	}
	putchar('\n');
}

int main(int argc, char *argv[])
{
	cl_platform_id	platform_ids[32];
	cl_uint			platform_num;
	cl_int			i, c, rc;

	while ((c = getopt(argc, argv, "lp:d:")) != -1)
	{
		switch (c)
		{
			case 'l':
				only_list = 1;
				break;
			case 'p':
				only_platform = atoi(optarg);
				break;
			case 'd':
				only_device = atoi(optarg);
				break;
			default:
				fprintf(stderr,
						"usage: %s [-l] [-p <platform>] [-d <device>]\n",
						basename(argv[0]));
				return 1;
		}
	}

	rc = clGetPlatformIDs(lengthof(platform_ids),
						  platform_ids,
						  &platform_num);
	if (rc != CL_SUCCESS)
	{
		fprintf(stderr, "failed on clGetPlatformIDs (%s)",
				opencl_strerror(rc));
		return 1;
	}

	for (i=0; i < platform_num; i++)
	{
		if (only_platform < 0 || i + 1 == only_platform)
			dump_platform(i, platform_ids[i]);
	}
	return 0;
}
