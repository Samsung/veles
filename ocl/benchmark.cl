#include "defines.cl"

__kernel __attribute__((reqd_work_group_size(B_BLOCK_SIZE, A_BLOCK_SIZE, 1)))
void benchmark(__global dtype *A, __global dtype *B, __global dtype *C) {
  #define A_WIDTH SIZE
  #define B_WIDTH SIZE
  #define AB_COMMON SIZE

  #include "matrix_multiplication.cl"

  if (valid) {
    C[idx] = sum;
  }
}
