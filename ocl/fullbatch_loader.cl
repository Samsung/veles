#include "defines.cl"
#include "highlight.cl"


__kernel void fill_minibatch_data_labels(
    __global const original_data_dtype    /* IN */    *original_data,
    __global minibatch_data_dtype        /* OUT */    *minibatch_data,
    const int                             /* IN */    start_offset,
    const int                             /* IN */    count,
#if LABELS > 0
    __global const int                    /* IN */    *original_labels,
    __global int                         /* OUT */    *minibatch_labels,
#endif
    __global const int                    /* IN */    *shuffled_indices,
    __global int                         /* OUT */    *minibatch_indices) {

  int sample_number = get_global_id(0);
  int real_sample_number = sample_number < count ? shuffled_indices[start_offset + sample_number] : -1;

  int offs_in_sample = get_global_id(1);
  int offs_in_data = real_sample_number * SAMPLE_SIZE + offs_in_sample;
  int offs_in_minibatch = sample_number * SAMPLE_SIZE + offs_in_sample;

  minibatch_data[offs_in_minibatch] = sample_number < count ? (minibatch_data_dtype)original_data[offs_in_data] : 0;
#if LABELS > 0
  minibatch_labels[sample_number] = sample_number < count ? original_labels[real_sample_number] : -1;
#endif
  minibatch_indices[sample_number] = real_sample_number;
}


#if TARGET > 0
__kernel void fill_minibatch_target(
    __global const original_target_dtype    /* IN */    *original_target,
    __global minibatch_target_dtype        /* OUT */    *minibatch_target,
    const int                               /* IN */    start_offset,
    const int                               /* IN */    count,
    __global int                            /* IN */    *shuffled_indices) {

  int sample_number = get_global_id(0);
  int real_sample_number = sample_number < count ? shuffled_indices[start_offset + sample_number] : -1;

  int offs_in_sample = get_global_id(1);
  int offs_in_target = real_sample_number * TARGET_SIZE + offs_in_sample;
  int offs_in_minibatch = sample_number * TARGET_SIZE + offs_in_sample;

  minibatch_target[offs_in_minibatch] = sample_number < count ? (minibatch_target_dtype)original_target[offs_in_target] : 0;
}
#endif

