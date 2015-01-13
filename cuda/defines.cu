#ifndef _DEFINES_
#define _DEFINES_

#include <highlight.cuh>

#ifndef dtype
#define dtype int
#endif

#ifndef FLT_MAX
#define FLT_MAX 1E+37
#endif

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define SIGN(x) ((x) ? ((x) > 0 ? 1 : -1) : 0)

/**
 * @brief CUDA implementation of atomicAdd for double.
 *
 * CUDA already has following implementations:
 * - int atomicAdd(int* address, int val);
 * - unsigned int atomicAdd(unsigned int* address, unsigned int val);
 * - unsigned long long int atomicAdd(unsigned long long int* address,
 *                                    unsigned long long int val);
 * - float atomicAdd(float* address, float val);
 */
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull =
  (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val +
            __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#endif  // _DEFINES_
