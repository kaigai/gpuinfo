#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

static void
__ereport(const char *func_name, int lineno,
		  CUresult errcode, const char *msg)
{
	const char *err_name;
	const char *err_str;

	cuGetErrorName(errcode, &err_name);
	cuGetErrorString(errcode, &err_str);

	fprintf(stderr, "%s:%d %s (%s:%s)\n",
			func_name, lineno, msg, err_name, err_str);
	exit(0);
}
#define ereport(errcode,msg)						\
	__ereport(__FUNCTION__,__LINE__,(errcode),(msg))

#define lengthof(array) (sizeof (array) / sizeof ((array)[0]))

#define ATTR_INT			0
#define ATTR_BYTES			1
#define ATTR_KB				2
#define ATTR_MB				3
#define ATTR_KHZ			4
#define ATTR_COMPUTEMODE	5
#define ATTR_BOOL			6

static struct {
	CUdevice_attribute attnum;
	int   atttype;
	const char *attname;
} attr_catalog[] = {
	{CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
	 ATTR_INT, "Max # of threads per block"},
	{CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
	 ATTR_INT, "Max block dimension X"},
	{CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
	 ATTR_INT, "Max block dimension Y"},
	{CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
	 ATTR_INT, "Max block dimension Z"},
	{CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
	 ATTR_INT, "Max grid dimension X"},
	{CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
	 ATTR_INT, "Max grid dimension Y"},
	{CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
	 ATTR_INT, "Max grid dimension Z"},
	{CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
	 ATTR_BYTES, "Max shared memory per block in bytes"},
	{CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY,
	 ATTR_BYTES, "Total constant memory"},
	{CU_DEVICE_ATTRIBUTE_WARP_SIZE,
	 ATTR_INT, "Warp size"},
	{CU_DEVICE_ATTRIBUTE_MAX_PITCH,
	 ATTR_INT, "Max pitch"},
	{CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK,
	 ATTR_INT, "Max registers per block"},
	{CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
	 ATTR_KHZ, "Clock rate [kHZ]"},
	{CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
	 ATTR_INT, "Texture alignment"},
	{CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
	 ATTR_INT, "Number of multiprocessors"},
	{CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT,
	 ATTR_BOOL, "Has kernel execution timeout"},
	{CU_DEVICE_ATTRIBUTE_INTEGRATED,
	 ATTR_BOOL, "Host integrated memory"},
	{CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
	 ATTR_BOOL, "Host memory mapping to device"},
	{CU_DEVICE_ATTRIBUTE_COMPUTE_MODE,
	 ATTR_COMPUTEMODE, "Compute mode"},
	{CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT,
	 ATTR_INT, "Surface alignment"},
	{CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
	 ATTR_BOOL, "Concurrent kernels"},
	{CU_DEVICE_ATTRIBUTE_ECC_ENABLED,
	 ATTR_BOOL, "ECC memory is supported"},
	{CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,
	 ATTR_INT, "PCI Bus ID"},
	{CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
	 ATTR_INT, "PCI Device ID"},
	{CU_DEVICE_ATTRIBUTE_TCC_DRIVER,
	 ATTR_BOOL, "TCC driver model"},
	{CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
	 ATTR_KHZ, "Peak memory clock rate"},
	{CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
	 ATTR_INT, "Global memory bus width"},
	{CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
	 ATTR_INT, "L2 cache size"},
	{CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
	 ATTR_INT, "Max threads per multiprocessor"},
	{CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT,
	 ATTR_INT, "Number of asynchronous engines"},
	{CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING,
	 ATTR_BOOL, "Unified address space support"},
	{CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID,
	 ATTR_INT, "PCI domain ID"},
	{CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
	 ATTR_INT, "Compute Capability Major"},
	{CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
	 ATTR_INT, "Compute Capability Minor"},
	{CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED,
	 ATTR_BOOL, "Stream priorities supported"},
	{CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED,
	 ATTR_BOOL, "L1 cache on global memory"},
	{CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED,
	 ATTR_BOOL, "L1 cache on local memory"},
	{CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
	 ATTR_BYTES, "Max shared memory per multiprocessor"},
	{CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR,
	 ATTR_INT, "Max # of 32bit registers per multiprocessor"},
	{CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
	 ATTR_BOOL, "Can allocate managed memory"},
	{CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD,
	 ATTR_BOOL, "Device is on a multi-GPU board"},
	{CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID,
	 ATTR_INT, "Unique id of the device if multi-GPU board"},
	{CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
	 ATTR_INT, "Max threads per block"},
};

int
main(int argc, const char *argv[])
{
	CUdevice	device;
	CUresult	rc;
	int			i, j, count;
	const char *label;

	rc = cuInit(0);
	if (rc != CUDA_SUCCESS)
		ereport(rc, "failed on cuInit");

	rc = cuDeviceGetCount(&count);
	if (rc != CUDA_SUCCESS)
		ereport(rc, "failed on cuDeviceGetCount");

	for (i = 0; i < count; i++)
	{
		char	dev_name[256];
		size_t	dev_memsz;
		int		dev_prop;

		rc = cuDeviceGet(&device, i);
		if (rc != CUDA_SUCCESS)
			ereport(rc, "failed on cuDeviceGet");

		rc = cuDeviceGetName(dev_name, sizeof(dev_name), device);
		if (rc != CUDA_SUCCESS)
			ereport(rc, "failed on cuDeviceGetName");
		printf("device name: %s\n", dev_name);

		rc = cuDeviceTotalMem(&dev_memsz, device);
		if (rc != CUDA_SUCCESS)
			ereport(rc, "failed on cuDeviceTotalMem");
		printf("global memory size: %zuMB\n", dev_memsz >> 20);

		for (j=0; j < lengthof(attr_catalog); j++)
		{
			const char *attname = attr_catalog[j].attname;
			int			attnum = attr_catalog[j].attnum;
			int			atttype = attr_catalog[j].atttype;

			rc = cuDeviceGetAttribute(&dev_prop, attnum, device);
			if (rc != CUDA_SUCCESS)
				ereport(rc, "failed on cuDeviceGetAttribute");
			switch (atttype)
			{
				case ATTR_BYTES:
					printf("%s:  %d\n", attname, dev_prop);
					break;
				case ATTR_KB:
					printf("%s:  %dkB\n", attname, dev_prop);
					break;
				case ATTR_MB:
					printf("%s:  %dMB\n", attname, dev_prop);
                    break;
				case ATTR_KHZ:
					printf("%s:  %dkHZ\n", attname, dev_prop);
                    break;
				case ATTR_COMPUTEMODE:
					switch (dev_prop)
					{
						case CU_COMPUTEMODE_DEFAULT:
							label = "default";
							break;
						case CU_COMPUTEMODE_EXCLUSIVE:
							label = "exclusive";
							break;
						case CU_COMPUTEMODE_PROHIBITED:
							label = "prohibited";
							break;
						case CU_COMPUTEMODE_EXCLUSIVE_PROCESS:
							label = "exclusive process";
							break;
						default:
							label = "unknown";
							break;
					}
					printf("%s:  %s\n", attname, label);
					break;
				case ATTR_BOOL:
					printf("%s:  %s\n", attname, dev_prop ? "true" : "false");
					break;
				default:
					printf("%s:  %d\n", attname, dev_prop);
					break;
			}
		}
	}
	return 0;
}
