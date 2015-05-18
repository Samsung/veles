// This file is included from matrix_multiplication.cl

#if BLOCK_SUM_VECTORIZED == 0
  // Plain sumation
  __local dtype *_AS = &AS[as_start];
  __local dtype *_BS = &BS[bs_start];
  dtype block_sum = 0;
  #ifdef MULTIPLY
    #pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
      block_sum += MULTIPLY(_AS[k], _BS[k]);
    }
  #else
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
