#include "defines.cu"

/*
@brief Normalizes array of images according to mean and dispersion.
@param input Array of the images.
@param mean Image with the mean over the dataset.
@param rdisp Image with the 1.0 / dispersion over the dataset.
@param output Array of the output images.
@details output[:] = ((dtype)input[:] - (dtype)mean) * rdisp.
*/
extern "C"
__global__ void normalize_mean_disp(const input_type    /* IN */    *input,
                                    const mean_type     /* IN */    *mean,
                                    const dtype         /* IN */    *rdisp,
                                    dtype              /* OUT */    *output) {
  int offs_in_sample = blockIdx.x * blockDim.x + threadIdx.x;
  int offs = (blockIdx.y * blockDim.y + threadIdx.y) * SAMPLE_SIZE + offs_in_sample;
  output[offs] = ((dtype)input[offs] - (dtype)mean[offs_in_sample]) * rdisp[offs_in_sample];
}
