/** @brief Type of packing data in matrix. */
typedef enum { ROW, COL } PackingType;

/** @brief Structure for storing matrix parameters. */
typedef struct tag_matrix_param_t {
  int rows;
  int cols;
  PackingType packing;
} matrix_param_t;

/**
 * @brief Returns element in matrix using indexes (i, j) and packing type of
 *        its data.
 */
inline dtype get_element(__global const dtype* m, const matrix_param_t* param,
                         int i, int j) {
  return param->packing == ROW ? m[i * param->cols + j] :
                                 m[j * param->rows + i];
}

/**
 * @brief Multiples matrices using info of them structure.
 *
 * Result and its index in matrix (with ROW packing type) you can find in
 * variables <i>res</i> and <i>idx</i> respectively.
 */
bool matrix_mul(__global const dtype *A, const matrix_param_t *A_param,
                __global const dtype *B, const matrix_param_t *B_param,
                dtype *res, int *idx) {
  int gx = get_group_id(0);
  int lx = get_local_id(0);
  int gy = get_group_id(1);
  int ly = get_local_id(1);

  __local dtype AS[BLOCK_SIZE][BLOCK_SIZE];
  __local dtype BS[BLOCK_SIZE][BLOCK_SIZE];

  int i_A, j_A, i_B, j_B;
  *res = 0;
  for (int j = 0, i = 0; j < A_param->cols; j += BLOCK_SIZE, i += BLOCK_SIZE) {
    j_A = j + lx;
    i_A = BLOCK_SIZE * gy + ly;
    j_B = BLOCK_SIZE * gx + lx;
    i_B = i + ly;

    // if current positions in shared matrices AS and BS are
    // out of range then set 0
    AS[ly][lx] = (j_A < A_param->cols && i_A < A_param->rows) ?
        get_element(A, A_param, i_A, j_A) : 0;
    BS[ly][lx] = (j_B < B_param->cols && i_B < B_param->rows) ?
        get_element(B, B_param, i_B, j_B) : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; ++k) {
      *res += AS[ly][k] * BS[k][lx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  *idx = get_global_id(1) * B_param->cols + get_global_id(0);
  return (get_global_id(1) < A_param->rows) &&
         (get_global_id(0) < B_param->cols);
}
