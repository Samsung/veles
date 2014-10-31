// This file is included from matrix_multiplication.cl

#if BLOCK_SUM_VECTORIZED == 0
  // Plain sumation
  __local dtype *_AS = &AS[as_start];
  __local dtype *_BS = &BS[bs_start];
  #ifdef MULTIPLY
    dtype block_sum = 0;
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
      block_sum += MULTIPLY(_AS[k], _BS[k]);
    }
  #elif BLOCK_SIZE == 16
    // Manually use fma for AMD
    dtype block_sum = fma(_AS[15], _BS[15],
                      fma(_AS[14], _BS[14],
                      fma(_AS[13], _BS[13],
                      fma(_AS[12], _BS[12],
                      fma(_AS[11], _BS[11],
                      fma(_AS[10], _BS[10],
                      fma(_AS[9], _BS[9],
                      fma(_AS[8], _BS[8],
                      fma(_AS[7], _BS[7],
                      fma(_AS[6], _BS[6],
                      fma(_AS[5], _BS[5],
                      fma(_AS[4], _BS[4],
                      fma(_AS[3], _BS[3],
                      fma(_AS[2], _BS[2],
                      fma(_AS[1], _BS[1],
                      _AS[0] * _BS[0])))))))))))))));
  #else
    dtype block_sum = 0;
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
      block_sum += _AS[k] * _BS[k];
    }
  #endif
#else
  // Vector summation
  dtype block_sum = 0;
  #if BLOCK_SIZE % 4 == 0
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE / 4; k++) {
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
