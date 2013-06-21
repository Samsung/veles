"""
Created on Jun 3, 2013

Wavelet transform.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import inline
import os


c_code = inline.Inline()
c_code.sources.append(os.path.dirname(inline.__file__) + "/c/wavelet.c")
c_code.function_descriptions = {"transform": "f*f*iiiiiv"}
c_code.compile()


def transform(data, tmp, width, height, dest_wh, L=2, forward=1):
    """Do a 2D Daubechies wavelet transform.

    Parameters:
        data: numpy float32 array.
        tmp: numpy float32 array same size as data.
        width: data width.
        height: data height.
        dest_wh: destination width or height for low frequency.
        L: number of Daubechie's coefficients.
        forward: do forward or backward transform.

    Returns:
        0, and data will containt result of the transform,
        where tmp is for temporary storage.
    """
    global c_code
    return c_code.execute("transform", data, tmp, width, height,
                          dest_wh, L, forward)
