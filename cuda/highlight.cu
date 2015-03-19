#ifndef HIGHTLIGHTING_CUH_
#define HIGHTLIGHTING_CUH_

#ifdef __CDT_PARSER__

#define __launch_bounds__(x)
#define __restrict__
#define __device__
#define __global__
#define __shared__
#define __constant__

struct int2 { int x; int y; };
struct uint2 { unsigned int x; unsigned int y; };
struct long2 { long x; long y; };
struct ulong2 { long x; long y; };
struct float2 { float x; float y; };

struct int3 { int x; int y; int z; };
struct uint3 { unsigned int x; unsigned int y; unsigned int z; };
struct long3 { long x; long y; long z; };
struct ulong3 { long x; long y; long z; };
struct float3 { float x; float y; float z; };

struct int4 { int x; int y; int z; int w; };
struct uint4 { unsigned int x; unsigned int y; unsigned int z; unsigned int w; };
struct float4 { float x; float y; float z; float w; };

struct dim3 { unsigned int x; unsigned int y; unsigned int z; };
typedef dim3 gridDim;
typedef dim3 blockDim;
typedef uint3 blockIdx;
typedef uint3 threadIdx;

typedef int warpSize;

void __syncthreads();

#endif  // __CDT_PARSER__

#endif  // HIGHTLIGHTING_CUH_
