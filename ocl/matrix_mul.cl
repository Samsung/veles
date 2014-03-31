  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int cur_row = get_global_id(1);
  int cur_col = get_global_id(0);

  __local dtype AS[BLOCK_SIZE][BLOCK_SIZE];
  __local dtype BS[BLOCK_SIZE][BLOCK_SIZE];

// pre-calculates offsets to get matrix elements faster
#ifdef A_COLS_DATA_PACK
  int A_offset = lx * A_ROWS + cur_row;
  int A_shift = BLOCK_SIZE * A_ROWS;
#else
  int A_offset = cur_row * A_COLS + lx;
  int A_shift = BLOCK_SIZE;
#endif  // A_COLS_DATA_PACK
#ifdef B_COLS_DATA_PACK
  int B_offset = cur_col * B_ROWS + ly;
  int B_shift = BLOCK_SIZE;
#else
  int B_offset = ly * B_COLS + cur_col;
  int B_shift = BLOCK_SIZE * B_COLS;
#endif  // B_COLS_DATA_PACK

// pre-calculates number of blocks in the matrix A row
#if (A_COLS % BLOCK_SIZE) == 0
  #define MAX_ITER (A_COLS / BLOCK_SIZE)
#else
  #define MAX_ITER (A_COLS / BLOCK_SIZE + 1)
#endif

  bool row_in_range = cur_row < A_ROWS;
  bool col_in_range = cur_col < B_COLS;

  for (int A_idx = A_offset, B_idx = B_offset, iter_num = 0, idx1 = lx, idx2 = ly;
       iter_num < MAX_ITER;
       A_idx += A_shift, B_idx += B_shift, ++iter_num, idx1 += BLOCK_SIZE, idx2 += BLOCK_SIZE) {
    // if current positions in shared matrices AS and BS are
    // out of range then set 0
    AS[ly][lx] = (row_in_range && idx1 < A_COLS) ? A[A_idx] : 0;
    BS[ly][lx] = (col_in_range && idx2 < B_ROWS) ? B[B_idx] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      res += AS[ly][k] * BS[k][lx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  idx = get_global_id(1) * B_COLS + get_global_id(0);
  bool is_in_range =  (get_global_id(1) < A_ROWS) && (get_global_id(0) < B_COLS);
