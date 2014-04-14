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

cl_mem clCreateBuffer(cl_context context,
					  cl_mem_flags flags,
					  size_t size,
					  void *host_ptr,
					  cl_int *errcode_ret)
{
	static cl_mem (*p_clCreateBuffer)(
		cl_context context,
		cl_mem_flags flags,
		size_t size,
		void *host_ptr,
		cl_int *errcode_ret) = NULL;

	if (!p_clCreateBuffer)
		p_clCreateBuffer = get_opencl_function("clCreateBuffer");

	return (*p_clCreateBuffer)(context,
							   flags,
							   size,
							   host_ptr,
							   errcode_ret);
}

cl_int clEnqueueReadBuffer(cl_command_queue command_queue,
						   cl_mem buffer,
						   cl_bool blocking_read,
						   size_t offset,
						   size_t size,
						   void *ptr,
						   cl_uint num_events_in_wait_list,
						   const cl_event *event_wait_list,
						   cl_event *event)
{
	static cl_int (*p_clEnqueueReadBuffer)(
		cl_command_queue command_queue,
		cl_mem buffer,
		cl_bool blocking_read,
		size_t offset,
		size_t size,
		void *ptr,
		cl_uint num_events_in_wait_list,
		const cl_event *event_wait_list,
		cl_event *event) = NULL;

	if (!p_clEnqueueReadBuffer)
		p_clEnqueueReadBuffer = get_opencl_function("clEnqueueReadBuffer");

	return (*p_clEnqueueReadBuffer)(command_queue,
									buffer,
									blocking_read,
									offset,
									size,
									ptr,
									num_events_in_wait_list,
									event_wait_list,
									event);
}

cl_int clEnqueueWriteBuffer(cl_command_queue command_queue,
							cl_mem buffer,
							cl_bool blocking_write,
							size_t offset,
							size_t size,
							const void *ptr,
							cl_uint num_events_in_wait_list,
							const cl_event *event_wait_list,
							cl_event *event)
{
	static cl_int (*p_clEnqueueWriteBuffer)(
		cl_command_queue command_queue,
		cl_mem buffer,
		cl_bool blocking_write,
		size_t offset,
		size_t size,
		const void *ptr,
		cl_uint num_events_in_wait_list,
		const cl_event *event_wait_list,
		cl_event *event) = NULL;

	if (!p_clEnqueueWriteBuffer)
		p_clEnqueueWriteBuffer = get_opencl_function("clEnqueueWriteBuffer");

	return (*p_clEnqueueWriteBuffer)(command_queue,
									 buffer,
									 blocking_write,
									 offset,
									 size,
									 ptr,
									 num_events_in_wait_list,
									 event_wait_list,
									 event);
}

cl_int clReleaseMemObject(cl_mem memobj)
{
	static cl_int (*p_clReleaseMemObject)(cl_mem memobj) = NULL;

	if (!p_clReleaseMemObject)
		p_clReleaseMemObject = get_opencl_function("clReleaseMemObject");

	return (*p_clReleaseMemObject)(memobj);
}

cl_command_queue clCreateCommandQueue(cl_context context,
									  cl_device_id device,
									  cl_command_queue_properties properties,
									  cl_int *errcode_ret)
{
	static cl_command_queue (*p_clCreateCommandQueue)(
		cl_context context,
		cl_device_id device,
		cl_command_queue_properties properties,
		cl_int *errcode_ret) = NULL;

	if (!p_clCreateCommandQueue)
		p_clCreateCommandQueue = get_opencl_function("clCreateCommandQueue");

	return (*p_clCreateCommandQueue)(context,
									 device,
									 properties,
									 errcode_ret);
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue)
{
	static cl_int (*p_clReleaseCommandQueue)(
		cl_command_queue command_queue) = NULL;

	if (!p_clReleaseCommandQueue)
		p_clReleaseCommandQueue = get_opencl_function("clReleaseCommandQueue");

	return (*p_clReleaseCommandQueue)(command_queue);
}

cl_kernel clCreateKernel(cl_program  program,
						 const char *kernel_name,
						 cl_int *errcode_ret)
{
	static cl_kernel (*p_clCreateKernel)(
		cl_program  program,
		const char *kernel_name,
		cl_int *errcode_ret) = NULL;

	if (!p_clCreateKernel)
		p_clCreateKernel = get_opencl_function("clCreateKernel");
	return (*p_clCreateKernel)(program, kernel_name, errcode_ret);
}

cl_int clSetKernelArg(cl_kernel kernel,
					  cl_uint arg_index,
					  size_t arg_size,
					  const void *arg_value)
{
	static cl_int (*p_clSetKernelArg)(
		cl_kernel kernel,
		cl_uint arg_index,
		size_t arg_size,
		const void *arg_value) = NULL;

	if (!p_clSetKernelArg)
		p_clSetKernelArg = get_opencl_function("clSetKernelArg");
	return (*p_clSetKernelArg)(kernel, arg_index, arg_size, arg_value);
}

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue,
							  cl_kernel kernel,
							  cl_uint work_dim,
							  const size_t *global_work_offset,
							  const size_t *global_work_size,
							  const size_t *local_work_size,
							  cl_uint num_events_in_wait_list,
							  const cl_event *event_wait_list,
							  cl_event *event)
{
	static cl_int (*p_clEnqueueNDRangeKernel)(
		cl_command_queue command_queue,
		cl_kernel kernel,
		cl_uint work_dim,
		const size_t *global_work_offset,
		const size_t *global_work_size,
		const size_t *local_work_size,
		cl_uint num_events_in_wait_list,
		const cl_event *event_wait_list,
		cl_event *event) = NULL;

	if (!p_clEnqueueNDRangeKernel)
		p_clEnqueueNDRangeKernel
			= get_opencl_function("clEnqueueNDRangeKernel");

	return (*p_clEnqueueNDRangeKernel)(command_queue,
									   kernel,
									   work_dim,
									   global_work_offset,
									   global_work_size,
									   local_work_size,
									   num_events_in_wait_list,
									   event_wait_list,
									   event);
}

cl_int clReleaseKernel(cl_kernel kernel)
{
	static cl_int (*p_clReleaseKernel)(cl_kernel kernel) = NULL;

	if (!p_clReleaseKernel)
		p_clReleaseKernel = get_opencl_function("clReleaseKernel");
	return (*p_clReleaseKernel)(kernel);
}

cl_int clSetEventCallback(cl_event event,
						  cl_int  command_exec_callback_type ,
						  void (CL_CALLBACK  *pfn_event_notify)(
							  cl_event event,
							  cl_int event_command_exec_status,
							  void *user_data),
						   void *user_data)
{
	static (*p_clSetEventCallback)(
		cl_event event,
		cl_int  command_exec_callback_type ,
		void (CL_CALLBACK  *pfn_event_notify)(
			cl_event event,
			cl_int event_command_exec_status,
			void *user_data),
		void *user_data) = NULL;

	if (!p_clSetEventCallback)
		p_clSetEventCallback = get_opencl_function("clSetEventCallback");
	return (*p_clSetEventCallback)(event, command_exec_callback_type,
								   pfn_event_notify, user_data);
}

cl_int clWaitForEvents(cl_uint num_events,
					   const cl_event *event_list)
{
	static cl_int (*p_clWaitForEvents)(
		cl_uint num_events,
		const cl_event *event_list) = NULL;

	if (!p_clWaitForEvents)
		p_clWaitForEvents = get_opencl_function("clWaitForEvents");

	return (*p_clWaitForEvents)(num_events, event_list);
}

cl_int clReleaseEvent(cl_event event)
{
	static cl_int (*p_clReleaseEvent)(
		cl_event event) = NULL;
	if (!p_clReleaseEvent)
		p_clReleaseEvent = get_opencl_function("clReleaseEvent");
	return (*p_clReleaseEvent)(event);
}

cl_int clFinish(cl_command_queue command_queue)
{
	static cl_int (*p_clFinish)(cl_command_queue command_queue) = NULL;

	if (!p_clFinish)
		p_clFinish = get_opencl_function("clFinish");

	return (*p_clFinish)(command_queue);
}

const char *
opencl_strerror(cl_int errcode)
{
	switch (errcode)
	{
		case CL_SUCCESS:
			return "success";
		case CL_DEVICE_NOT_FOUND:
			return "device not found";
		case CL_DEVICE_NOT_AVAILABLE:
			return "device not available";
		case CL_COMPILER_NOT_AVAILABLE:
			return "compiler not available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return "memory object allocation failure";
		case CL_OUT_OF_RESOURCES:
			return "out of resources";
		case CL_OUT_OF_HOST_MEMORY:
			return "out of host memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE:
			return "profiling info not available";
		case CL_MEM_COPY_OVERLAP:
			return "memory copy overlap";
		case CL_IMAGE_FORMAT_MISMATCH:
			return "image format mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:
			return "image format not supported";
		case CL_BUILD_PROGRAM_FAILURE:
			return "build program failure";
		case CL_MAP_FAILURE:
			return "map failure";
		case CL_MISALIGNED_SUB_BUFFER_OFFSET:
			return "misaligned sub-buffer offset";
		case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
			return "execution status error for event in wait list";
		case CL_INVALID_VALUE:
			return "invalid value";
		case CL_INVALID_DEVICE_TYPE:
			return "invalid device type";
		case CL_INVALID_PLATFORM:
			return "invalid platform";
		case CL_INVALID_DEVICE:
			return "invalid device";
		case CL_INVALID_CONTEXT:
			return "invalid context";
		case CL_INVALID_QUEUE_PROPERTIES:
			return "invalid queue properties";
		case CL_INVALID_COMMAND_QUEUE:
			return "invalid command queue";
		case CL_INVALID_HOST_PTR:
			return "invalid host pointer";
		case CL_INVALID_MEM_OBJECT:
			return "invalid memory object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
			return "invalid image format descriptor";
		case CL_INVALID_IMAGE_SIZE:
			return "invalid image size";
		case CL_INVALID_SAMPLER:
			return "invalid sampler";
		case CL_INVALID_BINARY:
			return "invalid binary";
		case CL_INVALID_BUILD_OPTIONS:
			return "invalid build options";
		case CL_INVALID_PROGRAM:
			return "invalid program";
		case CL_INVALID_PROGRAM_EXECUTABLE:
			return "invalid program executable";
		case CL_INVALID_KERNEL_NAME:
			return "invalid kernel name";
		case CL_INVALID_KERNEL_DEFINITION:
			return "invalid kernel definition";
		case CL_INVALID_KERNEL:
			return "invalid kernel";
		case CL_INVALID_ARG_INDEX:
			return "invalid argument index";
		case CL_INVALID_ARG_VALUE:
			return "invalid argument value";
		case CL_INVALID_ARG_SIZE:
			return "invalid argument size";
		case CL_INVALID_KERNEL_ARGS:
			return "invalid kernel arguments";
		case CL_INVALID_WORK_DIMENSION:
			return "invalid work dimension";
		case CL_INVALID_WORK_GROUP_SIZE:
			return "invalid group size";
		case CL_INVALID_WORK_ITEM_SIZE:
			return "invalid item size";
		case CL_INVALID_GLOBAL_OFFSET:
			return "invalid global offset";
		case CL_INVALID_EVENT_WAIT_LIST:
			return "invalid wait list";
		case CL_INVALID_EVENT:
			return "invalid event";
		case CL_INVALID_OPERATION:
			return "invalid operation";
		case CL_INVALID_GL_OBJECT:
			return "invalid GL object";
		case CL_INVALID_BUFFER_SIZE:
            return "invalid buffer size";
		case CL_INVALID_MIP_LEVEL:
			return "invalid MIP level";
		case CL_INVALID_GLOBAL_WORK_SIZE:
			return "invalid global work size";
		case CL_INVALID_PROPERTY:
			return "invalid property";
	}
	return "unknown error code";
}
