#include "defines.cl"
#include "highlight.cl"

/// @brief C = A * B * alpha + C * beta.
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE, BLOCK_SIZE, 1)))
void gemm(__global const dtype *A, __global const dtype *B, __global dtype *C,
          const dtype alpha, const dtype beta,
          const ulong offsetA, const ulong offsetB, const ulong offsetC) {
  A += (size_t)offsetA;
  B += (size_t)offsetB;
  C += (size_t)offsetC;
  #define STORE_OUTPUT "gemm.store_output.cl"
  #include "matrix_multiplication.cl"
}
