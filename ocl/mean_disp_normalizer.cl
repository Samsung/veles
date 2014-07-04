#include "defines.cl"
#include "highlight.cl"

/*
@brief Normalizes array of images according to mean and dispersion.
@param input Array of the images.
@param mean Image with the mean over the dataset.
@param rdisp Image with the 1.0 / dispersion over the dataset.
@param output Array of the output images.
@details output[:] = ((dtype)input[:] - (dtype)mean) * rdisp.
*/
__kernel
void normalize_mean_disp(__global const input_type    /* IN */    *input,
                         __global const mean_type     /* IN */    *mean,
                         __global const dtype         /* IN */    *rdisp,
                         __global dtype              /* OUT */    *output) {
  int offs_in_sample = get_global_id(0);
  int offs = get_global_id(1) * SAMPLE_SIZE + offs_in_sample;
  output[offs] = ((dtype)input[offs] - (dtype)mean[offs_in_sample]) * rdisp[offs_in_sample];
}
