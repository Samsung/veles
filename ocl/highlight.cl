#ifndef _HIGHLIGHT_CL_
#define _HIGHLIGHT_CL_

// This is a hack to fix highlighting of OpenCL *.cl files without errors
#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#endif  // __OPENCL_VERSION__

#endif  // _HIGHLIGHT_CL_
