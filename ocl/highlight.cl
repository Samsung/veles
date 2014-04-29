// This is a hack to fix highlighting of OpenCL *.cl files without errors
#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#endif  // __OPENCL_VERSION__
