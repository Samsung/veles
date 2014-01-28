#include "defines.cl"

__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void benchmark(__global c_dtype *A, __global c_dtype *B, __global c_dtype *C) {
  #define A_WIDTH SIZE
  #define B_WIDTH SIZE
  #define AB_COMMON SIZE

  #include "matrix_multiplication.cl"

  if (valid) {
    C[idx] = sum[0];
  }
}
