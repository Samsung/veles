/*
 * General definitions.
 */

#ifndef _DEFINES_
#define _DEFINES_

/// @brief Pragmas for features.
#if sizeof_dtype == 8
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif


/// @brief Minimum of the two values.
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


/// Definitions for complex numbers
#if sizeof_c_dtype == sizeof_dtype * 2

inline dtype c_re(c_dtype a) {
  return a.x;
}

inline c_dtype c_from_re(dtype re) {
  return (c_dtype)(re, 0);
}

inline c_dtype c_mul(c_dtype a, c_dtype b) {
  return (c_dtype)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline c_dtype c_div(c_dtype a, c_dtype b) {
  dtype d = b.x * b.x + b.y * b.y;
  return (c_dtype)((a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d);
}

inline c_dtype c_exp(c_dtype a) {
  dtype d = exp(a.x);
  return (c_dtype)(cos(a.y) * d, sin(a.y) * d);
}

inline c_dtype c_tanh(c_dtype a) {
  dtype s = sign(a.x);
  c_dtype z = (c_dtype)(a.x * s, a.y);
  c_dtype ze = c_exp(z * (dtype)-2.0);
  z = c_div((c_dtype)(1, 0) - ze, (c_dtype)(1, 0) + ze);
  z.x *= s;
  return z;
}

inline dtype c_norm2(c_dtype a) {
  return a.x * a.x + a.y * a.y;
}

inline dtype c_dist2(c_dtype a, c_dtype b) {
  return c_norm2(a - b);
}

inline dtype c_norm(c_dtype a) {
  return length(a);
}

inline c_dtype c_relu(c_dtype a) {
  // FIXME(a.kazantsev): add proper implementation.
  return (c_dtype)(a.x > 15 ? a.x : log(exp(a.x) + 1), a.y);
}

#elif sizeof_c_dtype == sizeof_dtype

#define c_re(a) (a)
#define c_from_re(re) ((dtype)(re))
#define c_mul(a, b) ((a) * (b))
#define c_div(a, b) ((a) / (b))
#define c_exp(a) exp(a)
#define c_tanh(a) tanh(a)
#define c_norm2(a) ((a) * (a))
#define c_dist2(a, b) (((a) - (b)) * ((a) - (b)))
#define c_norm(a) fabs(a)
#define c_relu(a) ((a) > 15 ? (a) : log(exp(a) + 1))

#else

#error Unsupported number type.

#endif


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
    int u = atom_cmpxchg((__global volatile int *)addr, v, w);
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


/// @brief atom_add for float2.
inline float2 atom_add_float2(__global float2 *addr, float2 vle) {
  __global float *a = (__global float*)addr;
  return (float2)(atom_add_float(a, vle.x), atom_add_float(a + 1, vle.y));
}


/// @brief atom_add for double2.
#if sizeof_dtype == 8
inline double2 atom_add_double2(__global double2 *addr, double2 vle) {
  __global double *a = (__global double*)addr;
  return (double2)(atom_add_double(a, vle.x), atom_add_double(a + 1, vle.y));
}
#endif


#if sizeof_c_dtype == sizeof_dtype * 2

#if sizeof_dtype == 4
#define ATOM_ADD(addr, vle) atom_add_float2(addr, vle)
#elif sizeof_dtype == 8
#define ATOM_ADD(addr, vle) atom_add_double2(addr, vle)
#else
#error Unsupported number type.
#endif

#elif sizeof_c_dtype == sizeof_dtype

#if sizeof_dtype == 4
#define ATOM_ADD(addr, vle) atom_add_float(addr, vle)
#elif sizeof_dtype == 8
#define ATOM_ADD(addr, vle) atom_add_double(addr, vle)
#else
#error Unsupported number type.
#endif

#else

#error Unsupported number type.

#endif

#endif


/// @brief Sets all elements of array to zero.
__kernel
void array_clear(__global c_dtype /*OUT*/ *arr) {
  arr[get_global_id(0)] = c_from_re(0);
}

#endif  // _DEFINES_
