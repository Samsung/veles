// Store the gemm result in the output matrix
sum *= alpha;
if (beta) {
  sum += C[idx] * beta;
}
C[idx] = sum;
