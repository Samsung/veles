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
#ifndef A_BLOCK_SIZE
#error "A_BLOCK_SIZE should be defined"
#endif
#ifndef B_BLOCK_SIZE
#error "B_BLOCK_SIZE should be defined"
#endif
#ifndef COMMON_BLOCK_SIZE
#define "COMMON_BLOCK_SIZE should be defined"
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
#ifdef A_ALIGNED
#undef A_ALIGNED
#endif
#ifdef B_ALIGNED
#undef B_ALIGNED
#endif
#ifdef COMMON_ALIGNED
#undef COMMON_ALIGNED
#endif
#ifdef valid
#undef valid
#endif
#ifdef DO_INC_X
#undef DO_INC_X
#endif

// Set default precision level
#ifndef PRECISION_LEVEL
#define PRECISION_LEVEL 0
#endif

// Choose to vectorize block sum or not
#ifndef VECTOR_OPT
#define VECTOR_OPT 0
#endif
#if (VECTOR_OPT <= 0) || (COMMON_BLOCK_SIZE % 4 != 0)
#define BLOCK_SUM_VECTORIZED 0
#else
#ifndef MULTIPLY
#define BLOCK_SUM_VECTORIZED 1
#else
#define BLOCK_SUM_VECTORIZED 0
#endif
#endif

// Compute number of block at common matrix side
#if (AB_COMMON % COMMON_BLOCK_SIZE) == 0
#define N_BLOCKS (AB_COMMON / COMMON_BLOCK_SIZE)
#define COMMON_ALIGNED 1
#else
#define N_BLOCKS (AB_COMMON / COMMON_BLOCK_SIZE + 1)
#define COMMON_ALIGNED 0
#endif
#if (A_WIDTH % A_BLOCK_SIZE) == 0
#define A_ALIGNED 1
#else
#define A_ALIGNED 0
#endif
#if (B_WIDTH % B_BLOCK_SIZE) == 0
#define B_ALIGNED 1
#else
#define B_ALIGNED 0
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

  __local dtype AS[A_BLOCK_SIZE * COMMON_BLOCK_SIZE]; // shared submatrix of A
  __local dtype BS[B_BLOCK_SIZE * COMMON_BLOCK_SIZE]; // shared submatrix of B

  // Block index in matrix C, where the values will be put
  int bx = get_group_id(0); // from 0 to B_WIDTH / BLOCK_SIZE - 1
  int by = get_group_id(1); // from 0 to A_WIDTH / BLOCK_SIZE - 1

  // Thread index, each thread calculates one element of the resulted submatrix
  int tx = get_local_id(0); // from 0 to B_BLOCK_SIZE - 1
  int ty = get_local_id(1); // from 0 to A_BLOCK_SIZE - 1

// Here we will fill various offsets for different matrix configurations
#define A_LIMIT (A_WIDTH * AB_COMMON)
#ifdef A_COL
  // Block A will slide vertically
  #if COMMON_BLOCK_SIZE <= B_BLOCK_SIZE
    int as_idx = ty * COMMON_BLOCK_SIZE + tx;  // index in the local memory where to prefetch the data
    // Increment for a_offs
    #define A_INC_OFFS (A_WIDTH * COMMON_BLOCK_SIZE)
    // Increment for a_x
    #define A_INC_X 0
    #if COMMON_BLOCK_SIZE < B_BLOCK_SIZE
      bool as_valid = tx < COMMON_BLOCK_SIZE;
    #else
      #define as_valid 1
    #endif
  #else
    int as_idx_start = ty * COMMON_BLOCK_SIZE + tx;
    #define FULL_A_COUNT (COMMON_BLOCK_SIZE % B_BLOCK_SIZE == 0 ? COMMON_BLOCK_SIZE / B_BLOCK_SIZE - 1 : COMMON_BLOCK_SIZE / B_BLOCK_SIZE)
    int as_idx = as_idx_start + FULL_A_COUNT * B_BLOCK_SIZE;
    // Increment for a_offs inside full block & in the remaining part
    #define A_INC_FULL_OFFS (A_WIDTH * B_BLOCK_SIZE)
    #define A_INC_OFFS (A_WIDTH * (COMMON_BLOCK_SIZE - FULL_A_COUNT * B_BLOCK_SIZE))
    // Increment for a_x inside full block & in the remaining part
    #define A_INC_FULL_X 0
    #define A_INC_X 0
    #if COMMON_BLOCK_SIZE % B_BLOCK_SIZE != 0
      bool as_valid = tx < COMMON_BLOCK_SIZE % B_BLOCK_SIZE;
    #else
      #define as_valid 1
    #endif
  #endif
  int a_x = by * A_BLOCK_SIZE + ty;  // offset from the start of the row
  int a_offs = tx * A_WIDTH + a_x;  // offset to read from A for the current thread
  // Horizontal limit
  #define A_LIMIT_X A_WIDTH
#else
  // Block A will slide horizontally
  #if COMMON_BLOCK_SIZE <= B_BLOCK_SIZE
    int as_idx = ty * COMMON_BLOCK_SIZE + tx;  // index in the local memory where to prefetch the data
    // Increment for a_offs
    #define A_INC_OFFS COMMON_BLOCK_SIZE
    // Increment for a_x
    #define A_INC_X COMMON_BLOCK_SIZE
    #if COMMON_BLOCK_SIZE < B_BLOCK_SIZE
      bool as_valid = tx < COMMON_BLOCK_SIZE;
    #else
      #define as_valid 1
    #endif
  #else
    int as_idx_start = ty * COMMON_BLOCK_SIZE + tx;
    #define FULL_A_COUNT (COMMON_BLOCK_SIZE % B_BLOCK_SIZE == 0 ? COMMON_BLOCK_SIZE / B_BLOCK_SIZE - 1 : COMMON_BLOCK_SIZE / B_BLOCK_SIZE)
    int as_idx = as_idx_start + FULL_A_COUNT * B_BLOCK_SIZE;
    // Increment for a_offs inside full block & in the remaining part
    #define A_INC_FULL_OFFS B_BLOCK_SIZE
    #define A_INC_OFFS (COMMON_BLOCK_SIZE - FULL_A_COUNT * B_BLOCK_SIZE)
    // Increment for a_x inside full block & in the remaining part
    #define A_INC_FULL_X B_BLOCK_SIZE
    #define A_INC_X (COMMON_BLOCK_SIZE - FULL_A_COUNT * B_BLOCK_SIZE)
    #if COMMON_BLOCK_SIZE % B_BLOCK_SIZE != 0
      bool as_valid = tx < COMMON_BLOCK_SIZE % B_BLOCK_SIZE;
    #else
      #define as_valid 1
    #endif
  #endif
  int a_x = tx;  // offset from the start of the row
  int a_offs = (by * A_BLOCK_SIZE + ty) * AB_COMMON + a_x;  // offset to read from A for the current thread
  // Horizontal limit
  #define A_LIMIT_X AB_COMMON
#endif

#define B_LIMIT (B_WIDTH * AB_COMMON)
#ifdef B_COL
  // Block B will slide vertically
  #if COMMON_BLOCK_SIZE <= A_BLOCK_SIZE
    int bs_idx = tx * COMMON_BLOCK_SIZE + ty;  // index in the local memory where to prefetch the data
    // Increment for b_offs
    #define B_INC_OFFS (B_WIDTH * COMMON_BLOCK_SIZE)
    // Increment for b_x
    #define B_INC_X 0
    #if COMMON_BLOCK_SIZE < A_BLOCK_SIZE
      bool bs_valid = ty < COMMON_BLOCK_SIZE;
    #else
      #define bs_valid 1
    #endif
  #else
    int bs_idx_start = tx * COMMON_BLOCK_SIZE + ty;
    #define FULL_B_COUNT (COMMON_BLOCK_SIZE % A_BLOCK_SIZE == 0 ? COMMON_BLOCK_SIZE / A_BLOCK_SIZE - 1 : COMMON_BLOCK_SIZE / A_BLOCK_SIZE)
    int bs_idx = bs_idx_start + FULL_B_COUNT * A_BLOCK_SIZE;
    // Increment for b_offs inside full block & in the remaining part
    #define B_INC_FULL_OFFS (B_WIDTH * A_BLOCK_SIZE)
    #define B_INC_OFFS (B_WIDTH * (COMMON_BLOCK_SIZE - FULL_B_COUNT * A_BLOCK_SIZE))
    // Increment for b_x inside full block & in the remaining part
    #define B_INC_FULL_X 0
    #define B_INC_X 0
    #if COMMON_BLOCK_SIZE % A_BLOCK_SIZE != 0
      bool bs_valid = ty < COMMON_BLOCK_SIZE % A_BLOCK_SIZE;
    #else
      #define bs_valid 1
    #endif
  #endif
  int b_x = bx * B_BLOCK_SIZE + tx;  // offset from the start of the row
  int b_offs = ty * B_WIDTH + b_x;  // offset to read from B for the current thread
  // Horizontal limit
  #define B_LIMIT_X B_WIDTH
#else
  // Block B will slide horizontally
  #if COMMON_BLOCK_SIZE <= A_BLOCK_SIZE
    int bs_idx = tx * COMMON_BLOCK_SIZE + ty;  // index in the local memory where to prefetch the data
    // Increment for b_offs
    #define B_INC_OFFS COMMON_BLOCK_SIZE
    // Increment for b_x
    #define B_INC_X COMMON_BLOCK_SIZE
    #if COMMON_BLOCK_SIZE < A_BLOCK_SIZE
      bool bs_valid = ty < COMMON_BLOCK_SIZE;
    #else
      #define bs_valid 1
    #endif
  #else
    int bs_idx_start = tx * COMMON_BLOCK_SIZE + ty;
    #define FULL_B_COUNT (COMMON_BLOCK_SIZE % A_BLOCK_SIZE == 0 ? COMMON_BLOCK_SIZE / A_BLOCK_SIZE - 1 : COMMON_BLOCK_SIZE / A_BLOCK_SIZE)
    int bs_idx = bs_idx_start + FULL_B_COUNT * A_BLOCK_SIZE;
    // Increment for b_offs inside full block & in the remaining part
    #define B_INC_FULL_OFFS A_BLOCK_SIZE
    #define B_INC_OFFS (COMMON_BLOCK_SIZE - FULL_B_COUNT * A_BLOCK_SIZE)
    // Increment for b_x inside full block & in the remaining part
    #define B_INC_FULL_X A_BLOCK_SIZE
    #define B_INC_X (COMMON_BLOCK_SIZE - FULL_B_COUNT * A_BLOCK_SIZE)
    #if COMMON_BLOCK_SIZE % A_BLOCK_SIZE != 0
      bool bs_valid = ty < COMMON_BLOCK_SIZE % A_BLOCK_SIZE;
    #else
      #define bs_valid 1
    #endif
  #endif
  int b_x = ty;  // offset from the start of the row
  int b_offs = (bx * B_BLOCK_SIZE + tx) * AB_COMMON + b_x;  // offset to read from A for the current thread
  // Horizontal limit
  #define B_LIMIT_X AB_COMMON
#endif

// Start offsets for local thread dot product
#if BLOCK_SUM_VECTORIZED == 0
  int as_start = ty * COMMON_BLOCK_SIZE;
  int bs_start = tx * COMMON_BLOCK_SIZE;
#elif BLOCK_SIZE % 4 == 0
  int as_start = ty * (COMMON_BLOCK_SIZE / 4);
  int bs_start = tx * (COMMON_BLOCK_SIZE / 4);
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
  for (int i = 0; i < N_BLOCKS; i++, a_offs += A_INC_OFFS, b_offs += B_INC_OFFS) {
    #if COMMON_BLOCK_SIZE > B_BLOCK_SIZE
    // Loop through first complete blocks
    #pragma unroll
    for (int j = 0; j < FULL_A_COUNT; j++, a_offs += A_INC_FULL_OFFS) {
      #if COMMON_ALIGNED == 1
        #if A_ALIGNED == 1
          AS[as_idx_start + j * B_BLOCK_SIZE] = A_REAL_OFFS_VALID ? A[A_REAL_OFFS] : 0;
        #else
          #ifdef A_COL
            AS[as_idx_start + j * B_BLOCK_SIZE] = ((A_REAL_OFFS_VALID) && (a_x < A_LIMIT_X)) ? A[A_REAL_OFFS] : 0;
            a_x += A_INC_FULL_X;
          #else
            AS[as_idx_start + j * B_BLOCK_SIZE] = ((A_REAL_OFFS_VALID) && (a_offs < A_LIMIT)) ? A[A_REAL_OFFS] : 0;
          #endif
        #endif
      #else
        #if A_ALIGNED == 1
          #ifdef A_COL
            AS[as_idx_start + j * B_BLOCK_SIZE] = ((A_REAL_OFFS_VALID) && (a_offs < A_LIMIT)) ? A[A_REAL_OFFS] : 0;
          #else
            AS[as_idx_start + j * B_BLOCK_SIZE] = ((A_REAL_OFFS_VALID) && (a_x < A_LIMIT_X)) ? A[A_REAL_OFFS] : 0;
            a_x += A_INC_FULL_X;
          #endif
        #else
          AS[as_idx_start + j * B_BLOCK_SIZE] = ((A_REAL_OFFS_VALID) && (a_offs < A_LIMIT) && (a_x < A_LIMIT_X)) ? A[A_REAL_OFFS] : 0;
          a_x += A_INC_FULL_X;
        #endif
      #endif
    }
    #endif
    // Process the remaining part
    if (as_valid) {
      #if COMMON_ALIGNED == 1
        #if A_ALIGNED == 1
          AS[as_idx] = A_REAL_OFFS_VALID ? A[A_REAL_OFFS] : 0;
        #else
          #ifdef A_COL
            AS[as_idx] = ((A_REAL_OFFS_VALID) && (a_x < A_LIMIT_X)) ? A[A_REAL_OFFS] : 0;
            #define DO_INC_X
          #else
            AS[as_idx] = ((A_REAL_OFFS_VALID) && (a_offs < A_LIMIT)) ? A[A_REAL_OFFS] : 0;
          #endif
        #endif
      #else
        #if A_ALIGNED == 1
          #ifdef A_COL
            AS[as_idx] = ((A_REAL_OFFS_VALID) && (a_offs < A_LIMIT)) ? A[A_REAL_OFFS] : 0;
          #else
            AS[as_idx] = ((A_REAL_OFFS_VALID) && (a_x < A_LIMIT_X)) ? A[A_REAL_OFFS] : 0;
            #define DO_INC_X
          #endif
        #else
          AS[as_idx] = ((A_REAL_OFFS_VALID) && (a_offs < A_LIMIT) && (a_x < A_LIMIT_X)) ? A[A_REAL_OFFS] : 0;
          #define DO_INC_X
        #endif
      #endif
    }
    #ifdef DO_INC_X
    a_x += A_INC_X;
    #undef DO_INC_X
    #endif

    #if COMMON_BLOCK_SIZE > A_BLOCK_SIZE
    // Loop through first complete blocks
    #pragma unroll
    for (int j = 0; j < FULL_B_COUNT; j++, b_offs += B_INC_FULL_OFFS) {
      #if COMMON_ALIGNED == 1
        #if B_ALIGNED == 1
          BS[bs_idx_start + j * A_BLOCK_SIZE] = B_REAL_OFFS_VALID ? B[B_REAL_OFFS] : 0;
        #else
          #ifdef B_COL
            BS[bs_idx_start + j * A_BLOCK_SIZE] = ((B_REAL_OFFS_VALID) && (b_x < B_LIMIT_X)) ? B[B_REAL_OFFS] : 0;
            b_x += B_INC_FULL_X;
          #else
            BS[bs_idx_start + j * A_BLOCK_SIZE] = ((B_REAL_OFFS_VALID) && (b_offs < B_LIMIT)) ? B[B_REAL_OFFS] : 0;
          #endif
        #endif
      #else
        #if B_ALIGNED == 1
          #ifdef B_COL
            BS[bs_idx_start + j * A_BLOCK_SIZE] = ((B_REAL_OFFS_VALID) && (b_offs < B_LIMIT)) ? B[B_REAL_OFFS] : 0;
          #else
            BS[bs_idx_start + j * A_BLOCK_SIZE] = ((B_REAL_OFFS_VALID) && (b_x < B_LIMIT_X)) ? B[B_REAL_OFFS] : 0;
            b_x += B_INC_FULL_X;
          #endif
        #else
          BS[bs_idx_start + j * A_BLOCK_SIZE] = ((B_REAL_OFFS_VALID) && (b_offs < B_LIMIT) && (b_x < B_LIMIT_X)) ? B[B_REAL_OFFS] : 0;
          b_x += B_INC_FULL_X;
        #endif
      #endif
    }
    #endif
    // Process the remaining part
    if (bs_valid) {
      #if COMMON_ALIGNED == 1
        #if B_ALIGNED == 1
          BS[bs_idx] = B_REAL_OFFS_VALID ? B[B_REAL_OFFS] : 0;
        #else
          #ifdef B_COL
            BS[bs_idx] = ((B_REAL_OFFS_VALID) && (b_x < B_LIMIT_X)) ? B[B_REAL_OFFS] : 0;
            #define DO_INC_X
          #else
            BS[bs_idx] = ((B_REAL_OFFS_VALID) && (b_offs < B_LIMIT)) ? B[B_REAL_OFFS] : 0;
          #endif
        #endif
      #else
        #if B_ALIGNED == 1
          #ifdef B_COL
            BS[bs_idx] = ((B_REAL_OFFS_VALID) && (b_offs < B_LIMIT)) ? B[B_REAL_OFFS] : 0;
          #else
            BS[bs_idx] = ((B_REAL_OFFS_VALID) && (b_x < B_LIMIT_X)) ? B[B_REAL_OFFS] : 0;
            #define DO_INC_X
          #endif
        #else
          BS[bs_idx] = ((B_REAL_OFFS_VALID) && (b_offs < B_LIMIT) && (b_x < B_LIMIT_X)) ? B[B_REAL_OFFS] : 0;
          #define DO_INC_X
        #endif
      #endif
    }
    #ifdef DO_INC_X
    b_x += B_INC_X;
    #undef DO_INC_X
    #endif

    // ensure all shared loaded
    barrier(CLK_LOCAL_MEM_FENCE);

    #if BLOCK_SUM_VECTORIZED == 0
      // Plain sumation
      __local dtype *_AS = &AS[as_start];
      __local dtype *_BS = &BS[bs_start];
      #ifdef MULTIPLY
      dtype block_sum = 0;
      #pragma unroll
      for (int k = 0; k < COMMON_BLOCK_SIZE; k++) {
        block_sum += MULTIPLY(_AS[k], _BS[k]);
      }
      #else
      dtype block_sum = 0;
      #pragma unroll
      for (int k = 0; k < COMMON_BLOCK_SIZE; k++) {
        block_sum += _AS[k] * _BS[k];
      }
      #endif
    #else
      // Vector summation
      dtype block_sum = 0;
      #if COMMON_BLOCK_SIZE % 4 == 0
        #pragma unroll
        for (int k = 0; k < COMMON_BLOCK_SIZE / 4; k++) {
          block_sum += dot(vload4(as_start + k, AS), vload4(bs_start + k, BS));
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

#if A_ALIGNED == 1
  #if B_ALIGNED == 1
    #define valid 1
  #else
    bool valid = (get_global_id(0) < B_WIDTH);
  #endif
#else
  #if B_ALIGNED == 1
    bool valid = (get_global_id(1) < A_WIDTH);
  #else
    bool valid = (get_global_id(1) < A_WIDTH) && (get_global_id(0) < B_WIDTH);
  #endif
#endif

// Undefine defined values
#ifdef as_valid
#undef as_valid
#endif
#ifdef bs_valid
#undef bs_valid
#endif
#undef A_LIMIT
#undef A_LIMIT_X
#ifdef A_INC_FULL_X
#undef A_INC_FULL_X
#endif
#undef A_INC_X
#undef A_INC_OFFS
#ifdef A_INC_FULL_OFFS
#undef A_INC_FULL_OFFS
#endif
#undef A_REAL_OFFS
#undef A_REAL_OFFS_VALID
#ifdef FULL_A_COUNT
#undef FULL_A_COUNT
#endif
#undef B_LIMIT
#undef B_LIMIT_X
#undef B_INC_X
#ifdef B_INC_FULL_X
#undef B_INC_FULL_X
#endif
#undef B_INC_OFFS
#ifdef B_INC_FULL_OFFS
#undef B_INC_FULL_OFFS
#endif
#undef B_REAL_OFFS
#undef B_REAL_OFFS_VALID
#ifdef FULL_B_COUNT
#undef FULL_B_COUNT
#endif
#undef N_BLOCKS

// The source for matrix multiplication ends here.
