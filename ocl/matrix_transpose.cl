#ifndef BLOCK_SIZE
#error "BLOCK_SIZE MUST be defined, 16 is a good value"
#endif

/// @brief Transposes the matrix.
/// @param input matrix to transpose.
/// @param output result of the transposition (cannot be the same as input).
/// @param width width of the matrix.
/// @param height height of the matrix.
/// @details global_size = [roundup(width, BLOCK_SIZE), roundup(height, BLOCK_SIZE)]
///          local_size = [BLOCK_SIZE, BLOCK_SIZE]. 
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void transpose(__global const dtype    /* IN */    *input,
               __global dtype         /* OUT */    *output,
               const int               /* IN */    width,
               const int               /* IN */    height) {
  __local dtype AS[BLOCK_SIZE * (BLOCK_SIZE + 1)];

  int bx = get_group_id(0);
  int by = get_group_id(1);
  int tx = get_local_id(0);
  int ty = get_local_id(1);
  int x = get_global_id(0);
  int y = get_global_id(1);

  bool a_valid = (x < width) && (y < height);
  int b_row = bx * BLOCK_SIZE + ty;
  int b_col = by * BLOCK_SIZE + tx;
  int a_offs = y * width + x;
  bool b_valid = (b_row < width) && (b_col < height);
  int shared_offs = ty * (BLOCK_SIZE + 1) + tx;
  int b_offs = b_row * height + b_col;
  
  AS[shared_offs] = (a_valid) ? input[a_offs] : 0;

  barrier(CLK_LOCAL_MEM_FENCE);

  if (b_valid) {
    output[b_offs] = AS[tx * (BLOCK_SIZE + 1) + ty];
  }
}
