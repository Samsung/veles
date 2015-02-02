#ifndef SAMPLE_SIZE
#error "SAMPLE_SIZE should be defined"
#endif
#ifndef MAX_MINIBATCH_SIZE
#error "MAX_MINIBATCH_SIZE should be defined"
#endif


extern "C"
__global__ void fill_minibatch_data_labels(
    const original_data_dtype    *original_data,
    minibatch_data_dtype         *minibatch_data,
    const int                    start_offset,
    const int                    count,
#if LABELS > 0
    const int                    *original_labels,
    int                          *minibatch_labels,
#endif
    const int                    *shuffled_indices,
    int                          *minibatch_indices) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int sample_number = idx / SAMPLE_SIZE;
  int real_sample_number = sample_number < count ? shuffled_indices[start_offset + sample_number] : -1;

  int offs_in_sample = idx % SAMPLE_SIZE;
  int offs_in_data = real_sample_number * SAMPLE_SIZE + offs_in_sample;
  int offs_in_minibatch = sample_number * SAMPLE_SIZE + offs_in_sample;

  if (idx < (MAX_MINIBATCH_SIZE * SAMPLE_SIZE)) {
    minibatch_data[offs_in_minibatch] = sample_number < count ? (minibatch_data_dtype)original_data[offs_in_data] : 0;
    #if LABELS > 0
      minibatch_labels[sample_number] = sample_number < count ? original_labels[real_sample_number] : -1;
    #endif
    minibatch_indices[sample_number] = real_sample_number;
  }
}


#if TARGET > 0
extern "C"
__global__ void fill_minibatch_target(
    const original_target_dtype    *original_target,
    minibatch_target_dtype         *minibatch_target,
    const int                      start_offset,
    const int                      count,
    int                            *shuffled_indices) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int sample_number = idx / TARGET_SIZE;
  int real_sample_number = sample_number < count ? shuffled_indices[start_offset + sample_number] : -1;

  int offs_in_sample = idx % TARGET_SIZE;
  int offs_in_target = real_sample_number * TARGET_SIZE + offs_in_sample;
  int offs_in_minibatch = sample_number * TARGET_SIZE + offs_in_sample;

  if (idx < (MAX_MINIBATCH_SIZE * TARGET_SIZE)) {
    minibatch_target[offs_in_minibatch] = sample_number < count ? (minibatch_target_dtype)original_target[offs_in_target] : 0;
  }
}
#endif
