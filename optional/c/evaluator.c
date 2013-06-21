/*
 * C-code for evaluator.py
 * @author: Kazantsev Alexey <a.kazantsev@samsung.com>
*/

/*
 * Computes softmax errors.
 * Returns number of errors.
 */
int ev_softmax(dtype *y, int y_size, int y_aligned_size, dtype *err_y, int batch_size, int full_size,
               itype *labels, dtype threshold, dtype threshold_low, unsigned char *skipped,
               int *n_skipped, int *confusion_matrix)
{
 int n_ok = 0, n_skip = 0;
 int i, offs;
 for(i = 0, offs = 0; i < batch_size; i++, offs += y_aligned_size)
 {
  dtype m = y[offs];
  int im = 0;
  for(int j = 1; j < y_size; j++)
  {
   dtype vle = y[offs + j];
   if(vle > m)
   {
    m = vle;
    im = j;
   }
  }
  int ireal = labels[i];
  if(confusion_matrix)
   confusion_matrix[im * y_size + ireal]++;
  if(im == ireal)
  {
   n_ok++;
   if((y[im] > threshold) ||
      ((y[im] > threshold_low) && (skipped[i])))
   {
    // Set error to 0
    for(int j = 0; j < y_size; j++)
     err_y[offs + j] = 0;
    skipped[i] = 1;
    n_skip++;
    continue;
   }
  }
  skipped[i] = 0;
  // Compute gradient
  for(int j = 0; j < y_size; j++)
   err_y[offs + j] = y[offs + j];
  err_y[offs + ireal] = y[offs + ireal] - (dtype)1.0;
 }
 // Set errors for excessive samples to zero
 for(; i < full_size; i++, offs += y_aligned_size)
 {
  for(int j = 0; j < y_size; j++)
  {
   err_y[offs + j] = 0;
  }
 }
 *n_skipped = n_skip;
 return batch_size - n_ok;
}
