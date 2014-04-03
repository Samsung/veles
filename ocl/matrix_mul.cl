// Checks if number of columns in first matrix is equal to rows in second one
#if A_COLS != B_ROWS
  #error "Number of columns in first matrix is not equal to rows in second one"
#endif

#undef ALIGNED
// Defines ALIGNED if matrices are aligned with BLOCK_SIZE
#if (A_COLS % BLOCK_SIZE == 0) && (A_ROWS % BLOCK_SIZE == 0) && \
    (B_ROWS % BLOCK_SIZE == 0) && (B_COLS % BLOCK_SIZE == 0)
  #define ALIGNED
#endif

  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int cur_row = get_global_id(1);
  int cur_col = get_global_id(0);

// pre-calculates offsets to get matrix elements faster
#ifdef A_COLS_DATA_PACK
  int a_offset = lx * A_ROWS + cur_row;
  #define A_SHIFT (BLOCK_SIZE * A_ROWS)
#else
  int a_offset = cur_row * A_COLS + lx;
  #define A_SHIFT BLOCK_SIZE
#endif  // A_COLS_DATA_PACK
#ifdef B_COLS_DATA_PACK
  int b_offset = cur_col * B_ROWS + ly;
  #define B_SHIFT BLOCK_SIZE
#else
  int b_offset = ly * B_COLS + cur_col;
  #define B_SHIFT (BLOCK_SIZE * B_COLS)
#endif  // B_COLS_DATA_PACK

// pre-calculates number of blocks in the matrix A row
#if (A_COLS % BLOCK_SIZE) == 0
  #define MAX_ITER (A_COLS / BLOCK_SIZE)
#else
  #define MAX_ITER (A_COLS / BLOCK_SIZE + 1)
#endif

  bool row_in_range = cur_row < A_ROWS;
  bool col_in_range = cur_col < B_COLS;
  __local dtype AS[BLOCK_SIZE][BLOCK_SIZE];
  __local dtype BS[BLOCK_SIZE][BLOCK_SIZE];

  for (int a_idx = a_offset, b_idx = b_offset, iter_num = 0, idx1 = lx, idx2 = ly;
       iter_num < MAX_ITER;
       a_idx += A_SHIFT, b_idx += B_SHIFT, ++iter_num, idx1 += BLOCK_SIZE, idx2 += BLOCK_SIZE) {
#ifndef ALIGNED
    // if current positions in shared matrices AS and BS are
    // out of range then set 0
    AS[ly][lx] = (row_in_range && idx1 < A_COLS) ? A[a_idx] : 0;
    BS[ly][lx] = (col_in_range && idx2 < B_ROWS) ? B[b_idx] : 0;
#else
    AS[ly][lx] = A[a_idx];
    BS[ly][lx] = B[b_idx];
#endif

    barrier(CLK_LOCAL_MEM_FENCE);

#if defined(VECTOR_OPT) && (dtype == double || dtype == float)
#if (BLOCK_SIZE % 4 != 0)
  #error "BLOCK_SIZE should be multiple of 4 for Intel OpenCL platform"
#endif
#if dtype == double
  #define vec4_dtype double4
#else
  #define vec4_dtype float4
#endif
    // Optimisation for platforms where is fast dot product build-in function
    for (int k = 0; k < BLOCK_SIZE / 4; ++k) {
      res += dot((vec4_dtype) (AS[ly][0 + k * 4], AS[ly][1 + k * 4],
                               AS[ly][2 + k * 4], AS[ly][3 + k * 4]),
                 (vec4_dtype) (BS[0 + k * 4][lx], BS[1 + k * 4][lx],
                               BS[2 + k * 4][lx], BS[3 + k * 4][lx]));
    }
#else
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      res += AS[ly][k] * BS[k][lx];
    }
#endif

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  idx = get_global_id(1) * B_COLS + get_global_id(0);
  bool is_in_range =  (get_global_id(1) < A_ROWS) && (get_global_id(0) < B_COLS);
