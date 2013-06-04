"""
Created on Jun 3, 2013

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""
import inline
import os


c_code = inline.Inline()
c_code.sources.append(os.path.dirname(inline.__file__) + "/c/wavelet.c")
c_code.function_descriptions = {"transform": "f*f*iiiiii"}
c_code.compile()


def transform(data, tmp, width, height, dest_wh, L=2, forward=1):
    global c_code
    return c_code.execute("transform", data, tmp, width, height,
                          dest_wh, L, forward)
