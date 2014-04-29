/// @brief Define for reduce operation on matrix rows or columns.
/// @author Kazantsev Alexey <a.kazantsev@samsung.com>
/// @details Kernel should be defined as:
///          __kernel __attribute__((reqd_work_group_size(REDUCE_SIZE, 1, 1)))
///
///          Sizes should be declared externally (values are given for example):
///          #define REDUCE_SIZE 64
///          #define A_WIDTH 10
///          #define A_HEIGHT 100500
///
///          As well as Matricies:
///          #define A err_y
///
///          And summation by columns if neccessary (otherwise summation by rows is assumed):
///          #define A_COL
///
///          size_t WorkSize[2] = {A_WIDTH * REDUCE_SIZE} or {A_HEIGHT * REDUCE_SIZE} #ifdef A_COL
///          size_t LocalSize[2] = {REDUCE_SIZE}
///
///          The result will be in (sum + AS[0]), output offset will be in bx, write it in if (tx == 0) { ... }
  __local c_dtype AS[REDUCE_SIZE];

  int bx = get_group_id(0); // from 0 to number of resulting output elements
  int tx = get_local_id(0); // from 0 to BLOCK_SIZE - 1

  c_dtype sum = c_from_re(0);

  #ifdef A_COL
  int offs = bx + tx * A_WIDTH;
  #define ARRAY_SIZE A_HEIGHT
  #define OFFS (REDUCE_SIZE * A_WIDTH)
  #else
  int offs = bx * A_WIDTH + tx;
  #define ARRAY_SIZE A_WIDTH
  #define OFFS REDUCE_SIZE
  #endif
  for (int i = 0; i < ARRAY_SIZE / REDUCE_SIZE; i++, offs += OFFS) {
    sum += A[offs];
  }
  // Sum the remaining part
  #if (ARRAY_SIZE % REDUCE_SIZE) != 0
  if (tx < ARRAY_SIZE % REDUCE_SIZE)
    sum += A[offs];
  #endif

  AS[tx] = sum;
  // ensure all shared loaded
  barrier(CLK_LOCAL_MEM_FENCE);

  // Final summation
  sum = c_from_re(0);
  int n = MIN(ARRAY_SIZE, REDUCE_SIZE);
  while (n > 1) {
    sum += (n & 1) ? AS[n - 1] : c_from_re(0);
    n >>= 1;
    if (tx < n) {
      AS[tx] += AS[n + tx];
    }
    // ensure all shared summed
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  #undef OFFS
  #undef ARRAY_SIZE

  // The result will be in (sum + AS[0]), output offset will be in bx, write it in if (tx == 0) { ... }

/// Define for reduce operation on matrix rows or columns ends here.
