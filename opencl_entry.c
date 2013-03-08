/*
 * opencl_entry.c
 *
 * Entrypoint of OpenCL interfaces that should be resolved and linked
 * at run-time.
 *
 * --
 * Copyright 2013 (c) PG-Strom Development Team
 * Copyright 2011-2012 (c) KaiGai Kohei <kaigai@kaigai.gr.jp>
 *
 * This software is an extension of PostgreSQL; You can use, copy,
 * modify or distribute it under the terms of 'LICENSE' included
 * within this package.
 */
#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <CL/cl.h>
#include <CL/cl_ext.h>

/*
 * Init opencl stubs
 */
static void	   *opencl_library_handle = NULL;

void
opencl_entry_init(void)
{
	opencl_library_handle = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
	if (!opencl_library_handle)
	{
		fprintf(stderr, "could not open OpenCL library: %s\n",
				dlerror());
		exit(1);
	}
}

static void *
get_opencl_function(const char *func_name)
{
	void   *func_addr;

	assert(opencl_library_handle != NULL);

	func_addr = dlsym(opencl_library_handle, func_name);
	if (!func_addr)
	{
		fprintf(stderr, "could not find symbol \"%s\" : %s\n",
				func_name, dlerror());
		exit(1);
	}
	return func_addr;
}

/*
 * Query Platform Info
 */
cl_int
clGetPlatformIDs(cl_uint num_entries,
				 cl_platform_id *platforms,
				 cl_uint *num_platforms)
{
	static cl_int (*p_clGetPlatformIDs)(cl_uint num_entries,
										cl_platform_id *platforms,
										cl_uint *num_platforms) = NULL;
	if (!p_clGetPlatformIDs)
		p_clGetPlatformIDs = get_opencl_function("clGetPlatformIDs");

	return (*p_clGetPlatformIDs)(num_entries,
								 platforms,
								 num_platforms);
}

cl_int
clGetPlatformInfo(cl_platform_id platform,
				  cl_platform_info param_name,
				  size_t param_value_size,
				  void *param_value,
				  size_t *param_value_size_ret)
{
	static cl_int (*p_clGetPlatformInfo)(cl_platform_id platform,
										 cl_platform_info param_name,
										 size_t param_value_size,
										 void *param_value,
										 size_t *param_value_size_ret) = NULL;
	if (!p_clGetPlatformInfo)
		p_clGetPlatformInfo = get_opencl_function("clGetPlatformInfo");

	return (*p_clGetPlatformInfo)(platform,
								  param_name,
								  param_value_size,
								  param_value,
								  param_value_size_ret);
}

/*
 * Query Devices
 */
cl_int
clGetDeviceIDs(cl_platform_id platform,
			   cl_device_type device_type,
			   cl_uint num_entries,
			   cl_device_id *devices,
			   cl_uint *num_devices)
{
	static cl_int (*p_clGetDeviceIDs)(cl_platform_id platform,
									  cl_device_type device_type,
									  cl_uint num_entries,
									  cl_device_id *devices,
									  cl_uint *num_devices) = NULL;
	if (!p_clGetDeviceIDs)
		p_clGetDeviceIDs = get_opencl_function("clGetDeviceIDs");

	return (*p_clGetDeviceIDs)(platform,
							   device_type,
							   num_entries,
							   devices,
							   num_devices);
}

cl_int
clGetDeviceInfo(cl_device_id device,
				cl_device_info param_name,
				size_t param_value_size,
				void *param_value,
				size_t *param_value_size_ret)
{
	static cl_int (*p_clGetDeviceInfo)(cl_device_id device,
									   cl_device_info param_name,
									   size_t param_value_size,
									   void *param_value,
									   size_t *param_value_size_ret) = NULL;
	if (!p_clGetDeviceInfo)
		p_clGetDeviceInfo = get_opencl_function("clGetDeviceInfo");

	return (*p_clGetDeviceInfo)(device,
								param_name,
								param_value_size,
								param_value,
								param_value_size_ret);
}
