/// @brief Define for matrix multiplication.
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
///
///          C = A * B
///
///          We will calculate values for block of matrix C for each workgroup.
///
///          [AB_COMMON][A_WIDTH] * [B_WIDTH][AB_COMMON] = [A_WIDTH][B_WIDTH]
///          size_t WorkSize[2] = {B_WIDTH, A_WIDTH}
///          size_t LocalSize[2] = {BLOCK_SIZE, BLOCK_SIZE}
///
///          The resulting sum will be in "sum[0]",
///          index in the resulting matrix will be in "idx",
///          "valid" will be set to true if "idx" is valid.

#ifdef ALIGNED
#undef ALIGNED
#endif
#ifdef valid
#undef valid
#endif
#if (AB_COMMON % BLOCK_SIZE) == 0
#define N_BLOCKS (AB_COMMON / BLOCK_SIZE)
#if ((A_WIDTH % BLOCK_SIZE) == 0) && ((B_WIDTH % BLOCK_SIZE) == 0)
#define ALIGNED
#endif
#else
#define N_BLOCKS (AB_COMMON / BLOCK_SIZE + 1)
#endif
#if (N_BLOCKS <= 2) || (sizeof_dtype >= 8)
#define N_SUM 1
#elif N_BLOCKS <= 4
#define N_SUM 2
#elif N_BLOCKS <= 8
#define N_SUM 4
#elif N_BLOCKS <= 16
#define N_SUM 8
#elif (N_BLOCKS <= 32) || (sizeof_c_dtype > 8)
#define N_SUM 16
#else
#define N_SUM 32
#endif
#ifndef A_REAL_OFFS
#define A_REAL_OFFS a_offs
#endif
#ifndef B_REAL_OFFS
#define B_REAL_OFFS b_offs
#endif
#ifndef A_REAL_OFFS_VALID
#define A_REAL_OFFS_VALID 1
#endif
#ifndef B_REAL_OFFS_VALID
#define B_REAL_OFFS_VALID 1
#endif
#ifndef MULTIPLY
#define MULTIPLY c_mul
#endif

  __local c_dtype AS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of A
  __local c_dtype BS[BLOCK_SIZE][BLOCK_SIZE]; // shared submatrix of B

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
  #define A_OFFS A_WIDTH * BLOCK_SIZE
  #define A_INC_X 0
  #define A_LIMIT_X A_WIDTH
#else
  // Block will slide horizontally
  int a_x = tx;
  int a_offs = (by * BLOCK_SIZE + ty) * AB_COMMON + a_x;
  #define A_OFFS BLOCK_SIZE
  #define A_INC_X BLOCK_SIZE
  #define A_LIMIT_X AB_COMMON
#endif

#define B_LIMIT (B_WIDTH * AB_COMMON)
#ifdef B_COL
  // Block will slide vertically
  int b_x = bx * BLOCK_SIZE + tx;
  int b_offs = ty * B_WIDTH + b_x;
  #define B_OFFS B_WIDTH * BLOCK_SIZE
  #define B_INC_X 0
  #define B_LIMIT_X B_WIDTH
#else
  // Block will slide horizontally
  int b_x = tx;
  int b_offs = (bx * BLOCK_SIZE + ty) * AB_COMMON + b_x;
  #define B_OFFS BLOCK_SIZE
  #define B_INC_X BLOCK_SIZE
  #define B_LIMIT_X AB_COMMON
#endif

  c_dtype sum[N_SUM];
  #if N_SUM > 1
  for (int i_sum = 0; i_sum < N_SUM; i_sum++) {
    sum[i_sum] = c_from_re(0);
    for (int i = N_BLOCKS * i_sum / N_SUM; i < N_BLOCKS * (i_sum + 1) / N_SUM; i++,
         a_offs += A_OFFS, b_offs += B_OFFS) {
  #else
    #define i_sum 0
    sum[i_sum] = c_from_re(0);
    for (int i = 0; i < N_BLOCKS; i++, a_offs += A_OFFS, b_offs += B_OFFS) {
  #endif
      #ifdef ALIGNED
      AS[ty][tx] = A_REAL_OFFS_VALID ? A[A_REAL_OFFS] : 0;
      BS[ty][tx] = B_REAL_OFFS_VALID ? B[B_REAL_OFFS] : 0;
      #else
      AS[ty][tx] = ((A_REAL_OFFS_VALID) && (a_offs < A_LIMIT) && (a_x < A_LIMIT_X)) ? A[A_REAL_OFFS] : 0;
      BS[ty][tx] = ((B_REAL_OFFS_VALID) && (b_offs < B_LIMIT) && (b_x < B_LIMIT_X)) ? B[B_REAL_OFFS] : 0;
      a_x += A_INC_X;
      b_x += B_INC_X;
      #endif

      // ensure all shared loaded
      barrier(CLK_LOCAL_MEM_FENCE);

      #pragma unroll
      for (int k = 0; k < BLOCK_SIZE; k++)
      #ifdef B_COL
      #ifdef A_COL
        sum[i_sum] += MULTIPLY(AS[k][ty], BS[k][tx]);
      #else
        sum[i_sum] += MULTIPLY(AS[ty][k], BS[k][tx]);
      #endif
      #else
      #ifdef A_COL
        sum[i_sum] += MULTIPLY(AS[k][ty], BS[tx][k]);
      #else
        sum[i_sum] += MULTIPLY(AS[ty][k], BS[tx][k]);
      #endif
      #endif

      // ensure we can reload shared with new values
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  #if N_SUM > 1
  }
  for (int n_sum = N_SUM; n_sum > 2; n_sum >>= 1) {
    for (int i_sum = 0; i_sum < (n_sum >> 1); i_sum++)
      sum[i_sum] = sum[i_sum << 1] + sum[(i_sum << 1) + 1];
  }
  sum[0] += sum[1];
  #else
  #undef i_sum
  #endif

  int idx = get_global_id(1) * B_WIDTH + get_global_id(0);
#ifdef ALIGNED
  #define valid 1
#else
  int valid = (get_global_id(1) < A_WIDTH) && (get_global_id(0) < B_WIDTH);
#endif

  #undef A_LIMIT
  #undef A_LIMIT_X
  #undef A_INC_X
  #undef A_OFFS
  #undef A_REAL_OFFS
  #undef A_REAL_OFFS_VALID
  #undef B_LIMIT
  #undef B_LIMIT_X
  #undef B_INC_X
  #undef B_OFFS
  #undef B_REAL_OFFS
  #undef B_REAL_OFFS_VALID
  #undef N_BLOCKS
  #undef N_SUM
  #undef MULTIPLY

// The source for matrix multiplication ends here.
