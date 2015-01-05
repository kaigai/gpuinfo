/*
 * gputest.c - test module for OpenCL/CUDA functionalities
 */
#include "postgres.h"
#include "fmgr.h"
#include "miscadmin.h"
#include <CL/cl.h>

/* declarations */
extern Datum gputest_init_opencl(PG_FUNCTION_ARGS);
extern Datum gputest_cleanup_opencl(PG_FUNCTION_ARGS);
extern void  _PG_init(void);


Datum
gputest_init_opencl(PG_FUNCTION_ARGS)
{
	cl_int		rc;

	PG_RETURN_INT32(rc);
}
PG_FUNCTION_INFO_V1(gputest_init_opencl);

Datum
gputest_cleanup_opencl(PG_FUNCTION_ARGS)
{
	cl_int		rc;


	PG_RETURN_INT32(rc);
}
PG_FUNCTION_INFO_V1(gputest_cleanup_opencl);

void
_PG_init(void)
{
	if (!process_shared_preload_libraries_in_progress)
		elog(ERROR, "gputest must be loaded via shared_preload_libraries");








}
