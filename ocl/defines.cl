/*
 * General definitions.
 */

#ifndef _DEFINES_
#define _DEFINES_


/// @brief Pragmas for features.
#if (sizeof_dtype == 8) && (__OPENCL_VERSION__ < 120)
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif


/// @brief Minimum of the two values.
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


#ifdef USE_ATOMICS


/// @brief atom_add for float.
inline float atom_add_float(__global float *addr, float vle) {
  float sum = *addr;
  float oldsum;
  do {
    oldsum = sum;
    sum += vle;
    int v = *(int*)&oldsum;
    int w = *(int*)&sum;
    int u = atomic_cmpxchg((__global volatile int *)addr, v, w);
    sum = *(float*)&u;
  }
  while (sum != oldsum);
  return sum;
}


/// @brief atom_add for double.
#if sizeof_dtype == 8
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
inline double atom_add_double(__global double *addr, double vle) {
  double sum = *addr;
  double oldsum;
  do {
    oldsum = sum;
    sum += vle;
    long v = *(long*)&oldsum;
    long w = *(long*)&sum;
    long u = atom_cmpxchg((__global volatile long *)addr, v, w);
    sum = *(double*)&u;
  }
  while (sum != oldsum);
  return sum;
}
#endif


#if sizeof_dtype == 4
#define ATOM_ADD(addr, vle) atom_add_float(addr, vle)
#elif sizeof_dtype == 8
#define ATOM_ADD(addr, vle) atom_add_double(addr, vle)
#else
#error Unsupported number type.
#endif


#endif  // USE_ATOMICS


/// @brief Sets all elements of array to zero.
#define KERNEL_CLEAR(kernel_name, data_type) __kernel void kernel_name(__global data_type *arr) { arr[get_global_id(0)] = 0; }


#endif  // _DEFINES_
