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

cl_context
clCreateContext(const cl_context_properties *properties,
				cl_uint num_devices,
				const cl_device_id *devices,
				void (CL_CALLBACK *pfn_notify)(
					const char *errinfo,
					const void *private_info,
					size_t cb,
					void *user_data),
				void *user_data,
				cl_int *errcode_ret)
{
	static cl_context (*p_clCreateContext)(
		const cl_context_properties *properties,
	    cl_uint num_devices,
		const cl_device_id *devices,
		void (CL_CALLBACK *pfn_notify)(
			const char *errinfo,
			const void *private_info,
			size_t cb,
			void *user_data),
		void *user_data,
		cl_int *errcode_ret) = NULL;

	if (!p_clCreateContext)
		p_clCreateContext = get_opencl_function("clCreateContext");

	return (*p_clCreateContext)(properties,
								num_devices,
								devices,
								pfn_notify,
								user_data,
								errcode_ret);
}

cl_int
clReleaseContext(cl_context context)
{
	static cl_int (*p_clReleaseContext)(cl_context) = NULL;

	if (!p_clReleaseContext)
		p_clReleaseContext = get_opencl_function("clReleaseContext");

	return (*p_clReleaseContext)(context);
}

cl_program
clCreateProgramWithSource(cl_context context,
						  cl_uint count,
						  const char **strings,
						  const size_t *lengths,
						  cl_int *errcode_ret)
{
	static cl_program (*p_clCreateProgramWithSource)(
		cl_context context,
		cl_uint count,
		const char **strings,
		const size_t *lengths,
		cl_int *errcode_ret) = NULL;

	if (!p_clCreateProgramWithSource)
		p_clCreateProgramWithSource
			= get_opencl_function("clCreateProgramWithSource");

	return (*p_clCreateProgramWithSource)(context,
										  count,
										  strings,
										  lengths,
										  errcode_ret);
}

cl_int
clReleaseProgram(cl_program program)
{
	static cl_int (*p_clReleaseProgram)(cl_program program) = NULL;

	if (!p_clReleaseProgram)
		p_clReleaseProgram = get_opencl_function("clReleaseProgram");

	return (*p_clReleaseProgram)(program);
}

cl_int
clBuildProgram(cl_program program,
               cl_uint num_devices,
               const cl_device_id *device_list,
               const char *options,
               void (CL_CALLBACK *pfn_notify)(
                   cl_program program,
                   void *user_data),
               void *user_data)
{
	static cl_int (*p_clBuildProgram)(
		cl_program program,
		cl_uint num_devices,
		const cl_device_id *device_list,
		const char *options,
		void (CL_CALLBACK *pfn_notify)(
			cl_program program,
			void *user_data),
		void *user_data) = NULL;

	if (!p_clBuildProgram)
		p_clBuildProgram = get_opencl_function("clBuildProgram");

	return (*p_clBuildProgram)(program,
							   num_devices,
							   device_list,
							   options,
							   pfn_notify,
							   user_data);
}

cl_int
clGetProgramBuildInfo(cl_program program,
                      cl_device_id device,
                      cl_program_build_info param_name,
                      size_t param_value_size,
                      void *param_value,
                      size_t *param_value_size_ret)
{
	static cl_int (*p_clGetProgramBuildInfo)(
		cl_program program,
		cl_device_id device,
		cl_program_build_info param_name,
		size_t param_value_size,
		void *param_value,
		size_t *param_value_size_ret) = NULL;

	if (!p_clGetProgramBuildInfo)
		p_clGetProgramBuildInfo = get_opencl_function("clGetProgramBuildInfo");

	return (*p_clGetProgramBuildInfo)(program,
									  device,
									  param_name,
									  param_value_size,
									  param_value,
									  param_value_size_ret);
}
