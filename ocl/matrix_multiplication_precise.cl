/// @brief Matrix multiplication with precision improvements.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @details Kernel should be defined as:
///          __kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
///
///          Sizes should be declared externally (values are given for example):
///          #define BLOCK_SIZE 16
///          #define A_WIDTH 512
///          #define B_WIDTH 256
///          #define AB_COMMON 131072
///
///          As well as Matricies:
///          #define A err_y
///          #define B h
///
///          And column order if neccessary (otherwise row order is assumed):
///          #define A_COL
///          #define B_COL
///          Output will be accessed by rows,
///          swap matrices outside the kernel to get access by columns.
///
///          C = A * B
///
///          We will calculate values for block of matrix C for each workgroup.
///
///          [AB_COMMON][A_WIDTH] * [B_WIDTH][AB_COMMON] = [A_WIDTH][B_WIDTH]
///          global_size = [B_WIDTH, A_WIDTH]
///          local_size = [BLOCK_SIZE, BLOCK_SIZE]
///
///          If MULTIPLY(a, b) is defined, it will be used instead of "*",
///          do not forget to undef it later.
///
///          If VECTOR_OPT is defined, dot will be used instead of "*",
///          ignored if MULTIPLY is defined.
///
///          If PRECISION_LEVEL = 0, simple summation will be used.
///          If PRECISION_LEVEL = 1, Kahan summation will be used.
///          If PRECISION_LEVEL >= 2, most precise algorithm will be used,
///          on test matrix with 250000 common size and random numbers of the same magnitude:
///          float: 2.04 - 2.2 times slower, 2 decimal digits more precise,
///          double: 1.7 - 1.9 times slower, 2 decimal digits more precise.
///
///          To process an output of matrix multiplication, define STORE_OUTPUT as a file to be included,
///          and in it, "idx" will be an offset in target matrix and "sum" is the current sum.
///          Include may reside in a loop.

#include "matrix_multiplication_begin.cl"

  __local dtype AS[BLOCK_SIZE * BLOCK_SIZE]; // shared submatrix of A
  __local dtype BS[BLOCK_SIZE * BLOCK_SIZE]; // shared submatrix of B

  // Block index in matrix C, where the values will be put
  int bx = get_group_id(0); // from 0 to B_WIDTH / BLOCK_SIZE - 1
  int by = get_group_id(1); // from 0 to A_WIDTH / BLOCK_SIZE - 1

  // Thread index, each thread calculates one element of the resulted submatrix
  int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1
  int ty = get_local_id(1); // from 0 to BLOCK_SIZE - 1

#define A_LIMIT (A_WIDTH * AB_COMMON)
#ifdef A_COL
  // Block will slide vertically
  int a_x = by * BLOCK_SIZE + tx;
  int a_offs = ty * A_WIDTH + a_x;
  int as_idx = tx * BLOCK_SIZE + ty;
  #define A_OFFS A_WIDTH * BLOCK_SIZE
  #define A_INC_X 0
  #define A_LIMIT_X A_WIDTH
#else
  // Block will slide horizontally
  int a_x = tx;
  int a_offs = (by * BLOCK_SIZE + ty) * AB_COMMON + a_x;
  int as_idx = ty * BLOCK_SIZE + tx;
  #define A_OFFS BLOCK_SIZE
  #define A_INC_X BLOCK_SIZE
  #define A_LIMIT_X AB_COMMON
#endif

#define B_LIMIT (B_WIDTH * AB_COMMON)
#ifdef B_COL
  // Block will slide vertically
  int b_x = bx * BLOCK_SIZE + tx;
  int b_offs = ty * B_WIDTH + b_x;
  #ifdef A_COL
    #define bs_idx as_idx
  #else
    int bs_idx = tx * BLOCK_SIZE + ty;
  #endif
  #define B_OFFS B_WIDTH * BLOCK_SIZE
  #define B_INC_X 0
  #define B_LIMIT_X B_WIDTH
#else
  // Block will slide horizontally
  int b_x = tx;
  int b_offs = (bx * BLOCK_SIZE + ty) * AB_COMMON + b_x;
  #ifdef A_COL
    int bs_idx = ty * BLOCK_SIZE + tx;
  #else
    #define bs_idx as_idx
  #endif
  #define B_OFFS BLOCK_SIZE
  #define B_INC_X BLOCK_SIZE
  #define B_LIMIT_X AB_COMMON
#endif

#if BLOCK_SUM_VECTORIZED == 0
  int as_start = ty * BLOCK_SIZE;
  int bs_start = tx * BLOCK_SIZE;
#elif BLOCK_SIZE % 4 == 0
  int as_start = ty * (BLOCK_SIZE / 4);
  int bs_start = tx * (BLOCK_SIZE / 4);
#else
  #error "Control should not reach this point"
#endif

#if PRECISION_LEVEL == 0
  dtype sum = 0;
#elif PRECISION_LEVEL == 1
  dtype sum = 0, partial = 0;
#elif PRECISION_LEVEL >= 2
  dtype partials[32];
  int n_partials = 0;
#else
  #error "Unsupported PRECISION_LEVEL"
#endif
  dtype a_vle, b_vle;
  #ifdef ALIGNED
    a_vle = A_REAL_OFFS_VALID ? A[A_REAL_OFFS] : 0;
    b_vle = B_REAL_OFFS_VALID ? B[B_REAL_OFFS] : 0;
  #else
    a_vle = ((A_REAL_OFFS_VALID) && (a_offs < A_LIMIT) && (a_x < A_LIMIT_X)) ? A[A_REAL_OFFS] : 0;
    b_vle = ((B_REAL_OFFS_VALID) && (b_offs < B_LIMIT) && (b_x < B_LIMIT_X)) ? B[B_REAL_OFFS] : 0;
    a_x += A_INC_X;
    b_x += B_INC_X;
  #endif
  for (int i = 0; i < N_BLOCKS - 1; i++) {
    AS[as_idx] = a_vle;
    BS[bs_idx] = b_vle;
    a_offs += A_OFFS;
    b_offs += B_OFFS;

    // Ensure all shared loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    #ifdef ALIGNED
      a_vle = A_REAL_OFFS_VALID ? A[A_REAL_OFFS] : 0;
      b_vle = B_REAL_OFFS_VALID ? B[B_REAL_OFFS] : 0;
    #else
      a_vle = ((A_REAL_OFFS_VALID) && (a_offs < A_LIMIT) && (a_x < A_LIMIT_X)) ? A[A_REAL_OFFS] : 0;
      b_vle = ((B_REAL_OFFS_VALID) && (b_offs < B_LIMIT) && (b_x < B_LIMIT_X)) ? B[B_REAL_OFFS] : 0;
      a_x += A_INC_X;
      b_x += B_INC_X;
    #endif

    // Compute subsum
    #include "matrix_multiplication_subsum.cl"

    // Ensure we can reload shared with new values
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  AS[as_idx] = a_vle;
  BS[bs_idx] = b_vle;
  // Ensure all shared loaded
  barrier(CLK_LOCAL_MEM_FENCE);
  // Compute subsum
  #include "matrix_multiplication_subsum.cl"
  
  #if PRECISION_LEVEL >= 2
  dtype sum = 0;
  for (int k = 0; k < n_partials; k++) {
    sum += partials[k];
  }
  #endif

  int idx = get_global_id(1) * B_WIDTH + get_global_id(0);

#ifdef ALIGNED
  #include STORE_OUTPUT
#else
  if ((get_global_id(1) < A_WIDTH) && (get_global_id(0) < B_WIDTH)) {
    #include STORE_OUTPUT
  }
#endif

#include "matrix_multiplication_end.cl"
