import logging
import numpy


def normalize_linear(arr, scale=1.0):
    """Normalizes array to [-1, 1] in-place.
    """
    arr -= arr.min()
    mx = arr.max()
    if mx:
        arr /= mx * 0.5
        arr -= 1.0
    arr *= scale


def normalize_mean_disp(arr):
    mean = numpy.mean(arr)
    mi = numpy.min(arr)
    mx = numpy.max(arr)
    ds = max(mean - mi, mx - mean)
    arr -= mean
    if ds:
        arr /= ds


def normalize_exp(arr):
    arr -= arr.max()
    numpy.exp(arr, arr)
    smm = arr.sum()
    arr /= smm


def calculate_pointwise_normalization(arr):
    """Calculates coefficiets of pointwise dataset normalization to [-1, 1].
    """
    mul = numpy.zeros_like(arr[0])
    add = numpy.zeros_like(arr[0])

    mins = numpy.min(arr, 0)
    maxs = numpy.max(arr, 0)
    ds = maxs - mins
    zs = numpy.nonzero(ds)

    mul[zs] = 2.0
    mul[zs] /= ds[zs]

    mins *= mul
    add[zs] = -1.0
    add[zs] -= mins[zs]

    logging.getLogger("Loader").debug("%f %f %f %f" % (mul.min(), mul.max(),
                                                       add.min(), add.max()))

    return mul, add


def normalize_pointwise(arr):
    mul, add = calculate_pointwise_normalization(arr)
    arr *= mul
    arr += add


def normalize_image(a, yuv=False):
    """Normalizes numpy array to interval [0, 255].
    """
    aa = a.astype(numpy.float32)
    if aa.__array_interface__["data"][0] == a.__array_interface__["data"][0]:
        aa = aa.copy()
    aa -= aa.min()
    m = aa.max()
    if m:
        m /= 255.0
        aa /= m
    else:
        aa[:] = 127.5
    if yuv and len(aa.shape) == 3 and aa.shape[2] == 3:
        aaa = numpy.empty_like(aa)
        aaa[:, :, 0:1] = aa[:, :, 0:1] + (aa[:, :, 2:3] - 128) * 1.402
        aaa[:, :, 1:2] = (aa[:, :, 0:1] + (aa[:, :, 1:2] - 128) * (-0.34414) +
                          (aa[:, :, 2:3] - 128) * (-0.71414))
        aaa[:, :, 2:3] = aa[:, :, 0:1] + (aa[:, :, 1:2] - 128) * 1.772
        numpy.clip(aaa, 0.0, 255.0, aa)
    return aa.astype(numpy.uint8)
