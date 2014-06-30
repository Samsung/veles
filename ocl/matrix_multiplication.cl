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
///          The result will be in "sum", output offset in "idx" and
///          "valid" will be 1 when "idx" is valid and 0 otherwise.

// Check for required defines
#ifndef BLOCK_SIZE
#error "BLOCK_SIZE should be defined"
#endif
#ifndef A_WIDTH
#error "A_WIDTH should be defined"
#endif
#ifndef B_WIDTH
#error "B_WIDTH should be defined"
#endif
#ifndef AB_COMMON
#error "AB_COMMON should be defined"
#endif

// Support for multiple includes
#ifdef ALIGNED
#undef ALIGNED
#endif

#ifdef valid
#undef valid
#endif

// Set default precision level
#ifndef PRECISION_LEVEL
#define PRECISION_LEVEL 0
#endif

// Choose to vectorize block sum or not
#ifndef VECTOR_OPT
#define VECTOR_OPT 0
#endif
// TODO(a.kazantsev): implement vectorization for other block sizes.
#if (VECTOR_OPT <= 0) || ((BLOCK_SIZE % 4 != 0) && (BLOCK_SIZE % 2 != 0))
#define BLOCK_SUM_VECTORIZED 0
#else
#ifndef MULTIPLY
#define BLOCK_SUM_VECTORIZED 1
#else
#define BLOCK_SUM_VECTORIZED 0
#endif
#endif

// Compute number of block at common matrix side
#if (AB_COMMON % BLOCK_SIZE) == 0
#define N_BLOCKS (AB_COMMON / BLOCK_SIZE)
#if ((A_WIDTH % BLOCK_SIZE) == 0) && ((B_WIDTH % BLOCK_SIZE) == 0)
#define ALIGNED
#endif
#else
#define N_BLOCKS (AB_COMMON / BLOCK_SIZE + 1)
#endif

// Define default offsets
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
#else
#if BLOCK_SIZE % 4 == 0
  int as_start = ty * (BLOCK_SIZE / 4);
  int bs_start = tx * (BLOCK_SIZE / 4);
#elif BLOCK_SIZE % 2 == 0
  int as_start = ty * (BLOCK_SIZE / 2);
  int bs_start = tx * (BLOCK_SIZE / 2);
#else
  #error "Control should not reach this point"
#endif
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
  for (int i = 0; i < N_BLOCKS; i++, a_offs += A_OFFS, b_offs += B_OFFS) {
    #ifdef ALIGNED
    AS[as_idx] = A_REAL_OFFS_VALID ? A[A_REAL_OFFS] : 0;
    BS[bs_idx] = B_REAL_OFFS_VALID ? B[B_REAL_OFFS] : 0;
    #else
    AS[as_idx] = ((A_REAL_OFFS_VALID) && (a_offs < A_LIMIT) && (a_x < A_LIMIT_X)) ? A[A_REAL_OFFS] : 0;
    BS[bs_idx] = ((B_REAL_OFFS_VALID) && (b_offs < B_LIMIT) && (b_x < B_LIMIT_X)) ? B[B_REAL_OFFS] : 0;
    a_x += A_INC_X;
    b_x += B_INC_X;
    #endif

    // ensure all shared loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    #if BLOCK_SUM_VECTORIZED == 0
      // Plain sumation
      dtype block_sum = 0;
      #pragma unroll
      for (int k = 0; k < BLOCK_SIZE; k++) {
        #ifndef MULTIPLY
          block_sum += AS[as_start + k] * BS[bs_start + k];
        #else
          block_sum += MULTIPLY(AS[as_start + k], BS[bs_start + k]);
        #endif
      }
    #else
      // Vector summation
      dtype block_sum = 0;
      #if BLOCK_SIZE % 4 == 0
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE / 4; k++) {
          block_sum += dot(vload4(as_start + k, AS), vload4(bs_start + k, BS));
        }
      #elif BLOCK_SIZE % 2 == 0
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE / 2; k++) {
          block_sum += dot(vload2(as_start + k, AS), vload2(bs_start + k, BS));
        }
      #else
        #error "Control should not reach this point"
      #endif
    #endif

    #if PRECISION_LEVEL == 0
      // Simple summation
      sum += block_sum;
    #elif PRECISION_LEVEL == 1
      // Kahan summation
      volatile dtype y = block_sum - partial;
      dtype t = sum + y;
      partial = (t - sum) - y;
      sum = t;
    #elif PRECISION_LEVEL >= 2
      // Most precise summation
      int i = 0;
      #define x block_sum
      for (int k = 0; k < n_partials; k++) {
        dtype y = partials[k];
        if (fabs(x) < fabs(y)) {
          dtype t = x;
          x = y;
          y = t;
        }
        volatile dtype hi = x + y;
        dtype lo = y - (hi - x);
        if (lo) {
          partials[i++] = lo;
        }
        x = hi;
      }
      partials[i] = x;
      n_partials = i + 1;
      #undef x
    #endif

    // ensure we can reload shared with new values
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  #if PRECISION_LEVEL >= 2
  dtype sum = 0;
  for (int k = 0; k < n_partials; k++) {
    sum += partials[k];
  }
  #endif

  int idx = get_global_id(1) * B_WIDTH + get_global_id(0);

#ifdef ALIGNED
  #define valid 1
#else
  int valid = (get_global_id(1) < A_WIDTH) && (get_global_id(0) < B_WIDTH);
#endif

// Undefine defined values
#ifdef bs_idx
#undef bs_idx
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

// The source for matrix multiplication ends here.
