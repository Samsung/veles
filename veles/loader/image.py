# -*- coding: utf-8 -*-
"""
Created on Aug 14, 2013

Ontology of image loading classes.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
"""


from __future__ import division
from itertools import chain
from mimetypes import guess_type
import os
import re
import cv2
import numpy
from PIL import Image
from zope.interface import implementer, Interface

from veles.compat import from_none
import veles.error as error
from veles.loader.base import CLASS_NAME, TARGET, ILoader, Loader, \
    LoaderMSEMixin


MODE_COLOR_MAP = {
    "1": "GRAY",
    "L": "GRAY",
    "P": "RGB",
    "RGB": "RGB",
    "RGBA": "RGBA",
    "CMYK": "RGB",
    "YCbCr": "YCR_CB",
    "I": "GRAY",
    "F": "GRAY",
}

COLOR_CHANNELS_MAP = {
    "RGB": 3,
    "BGR": 3,
    "GRAY": 1,
    "HSV": 3,
    "YCR_CB": 3,
    "RGBA": 4,
    "BGRA": 4,
    "LAB": 3,
    "LUV": 3,
}


class IImageLoader(Interface):
    def get_image_label(key):
        """Retrieves label for the specified key.
        """

    def get_image_info(key):
        """
        Return a tuple (size, color space).
        Size must be in OpenCV order (first y, then x),
        color space must be supported by OpenCV (COLOR_*).
        """

    def get_image_data(key):
        """Return the image data associated with the specified key.
        """

    def get_keys(index):
        """
        Return a list of image keys to process for the specified class index.
        """


@implementer(ILoader)
class ImageLoader(Loader):
    """Base class for all image loaders. It is generally used for loading large
    datasets.

    Attributes:
        color_space: the color space to which to convert images. Can be any of
                     the values supported by OpenCV, e.g., GRAY or HSV.
        source_dtype: dtype to work with during various image operations.
        shape: image shape (tuple) - set after initialize().

     Must be overriden in child classes:
        get_image_label()
        get_image_info()
        get_image_data()
        get_keys()
    """

    def __init__(self, workflow, **kwargs):
        super(ImageLoader, self).__init__(workflow, **kwargs)
        self.color_space = kwargs.get("color_space", "RGB")
        self.source_dtype = numpy.float32
        self._uncropped_shape = tuple()
        self._has_labels = False
        self.class_keys = [[], [], []]
        self.verify_interface(IImageLoader)
        self._restored_from_pickle = False
        self.path_to_mean = kwargs.get("path_to_mean", None)
        self.add_sobel = kwargs.get("add_sobel", False)
        self.mirror = kwargs.get("mirror", False)  # True, False, "random"
        self.scale = kwargs.get("scale", 1.0)
        self.scale_maintain_aspect_ratio = kwargs.get(
            "scale_maintain_aspect_ratio", True)
        self.rotations = kwargs.get("rotations", (0.0,))  # radians
        self.background_image = kwargs.get("background_image", None)
        self.background_color = kwargs.get(
            "background_color", (0xff, 0x14, 0x93))
        self.crop = kwargs.get("crop", None)
        self.crop_number = kwargs.get("crop_number", 1)
        self._background = None
        self.mean_image = None

    def __setstate__(self, state):
        super(ImageLoader, self).__setstate__(state)
        self.info("Scanning for changes...")
        for keys in self.class_keys:
            for key in keys:
                size, _ = self.get_effective_image_info(key)
                if size != self.uncropped_shape:
                    raise error.BadFormatError(
                        "%s changed the effective size (now %s, was %s)" %
                        (key, size, self.uncropped_shape))
        self._restored_from_pickle = True

    @Loader.shape.getter
    def shape(self):
        """
        :return: Final cropped image shape.
        """
        if self.crop is not None:
            shape = self.crop
        else:
            shape = self.uncropped_shape
        if self.channels_number > 1:
            shape += (self.channels_number,)
        return shape

    @property
    def uncropped_shape(self):
        """
        :return: Uncropped (but scaled) image shape.
        """
        return self._uncropped_shape

    @uncropped_shape.setter
    def uncropped_shape(self, value):
        if value is None:
            raise ValueError("shape must not be None")
        if not isinstance(value, tuple):
            raise TypeError("shape must be a tuple (got %s)" % (value,))
        if len(value) not in (2, 3):
            raise ValueError("len(shape) must be equal to 2 or 3 (got %s)" %
                             (value,))
        for i, d in enumerate(value):
            if not isinstance(d, int):
                raise TypeError("shape[%d] is not an integer (= %s)" % (i, d))
            if d < 1:
                raise ValueError("shape[%d] < 1 (= %s)" % (i, d))
        if not isinstance(self.scale, tuple):
            self._uncropped_shape = tuple(
                int(value[i] * self.scale) for i in (0, 1))
        else:
            self.warning("Setting uncropped_shape is ignored: scale is %s" %
                         self.scale)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        if not isinstance(value, (float, tuple)):
            raise TypeError("scale must be either float or tuple of two ints"
                            " (got %s of type %s)" % (value, value.__class__))
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("scale must have length 2 (not %d in %s)" %
                                 (len(value), value))
            if not isinstance(value[0], int) or not isinstance(value[1], int):
                raise ValueError("scale must consist of integers (got %s)" %
                                 value)
            self._uncropped_shape = self.scale
        self._scale = value

    @property
    def crop(self):
        return self._crop

    @crop.setter
    def crop(self, value):
        if value is None:
            self._crop = None
            return
        if not isinstance(value, tuple):
            raise TypeError(
                "crop must be a tuple of 2 integers or floats (got %s)" %
                value)
        if len(value) != 2:
            raise ValueError("invalid crop length (got %d for %s), must be 2" %
                             (len(value), value))
        for i, val in enumerate(value):
            if not isinstance(val, (int, float)):
                raise TypeError(
                    "crop[%d] = %s is neither an integer nor a float" %
                    (i, val[i]))
            if isinstance(val, int) and val < 1:
                raise ValueError(
                    "crop[%d] = %s is out of range" % (i, val))
            if isinstance(val, float):
                if val <= 0 or val > 1:
                    raise ValueError(
                        "Out of range crop %s: %s" %
                        (("height", "width")[i], val))
        self._crop = value

    @property
    def crop_number(self):
        return self._crop_number

    @crop_number.setter
    def crop_number(self, value):
        if not isinstance(value, int):
            raise TypeError("crop_number must be an integer (got %s)" % value)
        if value < 1:
            raise ValueError(
                "crop_number must be greater than zero (got %d)" % value)
        if value > 1 and self.crop is None:
            raise ValueError(
                "crop parameter is None, refusing to set crop_number")
        self._crop_number = value

    @property
    def mirror(self):
        return self._mirror

    @mirror.setter
    def mirror(self, value):
        if value not in (False, True, "random"):
            raise ValueError(
                "mirror must be any of the following: False, True, \"random\"")
        self._mirror = value

    @property
    def rotations(self):
        return self._rotations

    @rotations.setter
    def rotations(self, value):
        if not isinstance(value, tuple):
            raise TypeError("rotations must be a tuple (got %s)" % value)
        for i, rot in enumerate(value):
            if not isinstance(rot, float):
                raise TypeError(
                    "rotations[%d] = %s is not a float" % (i, rot))
            if rot >= numpy.pi * 2:
                raise ValueError(
                    "rotations[%d] = %s is greater than 2π" % (i, rot))
        self._rotations = tuple(sorted(value))

    @property
    def samples_inflation(self):
        return (1 if self.mirror is not True else 2) * len(self.rotations) * \
            self.crop_number

    @property
    def background_image(self):
        return self._background_image

    @background_image.setter
    def background_image(self, value):
        if isinstance(value, str):
            with open(value, "rb") as fin:
                self.background_image = fin
        elif hasattr(value, "read") and hasattr(value, "seek"):
            self.background_image = numpy.array(Image.open(value))
        elif isinstance(value, numpy.ndarray):
            if value.shape != self.shape:
                raise error.BadFormatError(
                    "background_image's shape %s != sample's shape "
                    "%s" % (value.shape, self.shape))
            self._background_image = value
            if self.background_color is not None:
                self.warning(
                    "background_color = %s is ignored in favor of "
                    "background_image", self.background_color)
        elif value is None:
            self._background_image = None
        else:
            raise ValueError(
                "background_image must be any of the following: "
                "file name, file object, numpy array or None")

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, value):
        if value is None:
            self._background_color = None
            return
        if not isinstance(value, tuple):
            raise TypeError(
                "background_color must be a tuple (got %s)" % value)
        if len(value) != self.channels_number:
            raise ValueError(
                "background_color must have the same length as the number of "
                "channels = %d (got length %d for %s)" %
                (self.channels_number, len(value), value))
        for i, col in enumerate(value):
            if not isinstance(col, int):
                raise TypeError(
                    "background_color[%d] = %s is not an integer" % (i, col))
        if self.background_image is not None:
            self.warning(
                "background_color = %s is ignored in favor of "
                "background_image", self.background_color)
        self._background_color = value

    @property
    def has_labels(self):
        """
        This is set after initialize() (particularly, after load_data()).
        """
        return self._has_labels

    @property
    def channels_number(self):
        channels = COLOR_CHANNELS_MAP[self.color_space]
        if self.add_sobel:
            channels += 1
        return channels

    def get_effective_image_info(self, key):
        info = self.get_image_info(key)
        if self.scale == 1.0:
            return info
        if isinstance(self.scale, tuple):
            return self.scale, info[1]
        else:
            return tuple(int(info[0][i] * self.scale)
                         for i in (0, 1)), info[1]

    def preprocess_image(self, data, color, crop=True):
        if color != self.color_space:
            method = getattr(
                cv2, "COLOR_%s2%s" % (color, self.color_space), None)
            if method is None:
                aux_method = getattr(cv2, "COLOR_%s2BGR" % color)
                try:
                    data = cv2.cvtColor(data, aux_method)
                except cv2.error as e:
                    self.error("Failed to perform '%s' conversion", aux_method)
                    raise from_none(e)
                method = getattr(cv2, "COLOR_BGR2%s" % self.color_space)
            try:
                data = cv2.cvtColor(data, method)
            except cv2.error as e:
                self.error("Failed to perform '%s' conversion", method)
                raise from_none(e)

        if self.add_sobel:
            data = self.add_sobel_channel(data)
        if self.scale != 1.0:
            data = self.scale_image(data)
        if crop and self.crop is not None:
            data = self.crop_image(data)

        return data

    def scale_image(self, data):
        if self.scale_maintain_aspect_ratio:
            if data.shape[1] >= data.shape[0]:
                dst_width = self.uncropped_shape[:2][1]
                dst_height = int(numpy.round(
                    float(dst_width) * data.shape[0] / data.shape[1]))
            else:
                dst_height = self.uncropped_shape[:2][0]
                dst_width = int(numpy.round(
                    float(dst_height) * data.shape[1] / data.shape[0]))
            dst_x_min = int(
                numpy.round(
                    0.5 * (self.uncropped_shape[:2][1] - dst_width)))
            dst_y_min = int(
                numpy.round(
                    0.5 * (self.uncropped_shape[:2][0] - dst_height)))
            dst_x_max = dst_x_min + self.uncropped_shape[:2][1]
            dst_y_max = dst_y_min + self.uncropped_shape[:2][0]
            data = cv2.resize(
                data, (dst_width, dst_height),
                interpolation=cv2.INTER_CUBIC)
            sample = self.background_image.copy()
            sample[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = data
            data = sample.copy()
        else:
            cv2.resize(
                data, tuple(reversed(self.uncropped_shape[:2])),
                interpolation=cv2.INTER_CUBIC)
        return data

    def add_sobel_channel(self, data):
        if self.channels_number == 1:
            pass
        elif self.color_space in ("RGB", "BGR", "RGBA", "BGRA"):
            data = cv2.cvtColor(
                data, getattr(cv2, "COLOR_%s2GRAY" % self.color_space))
        elif self.color_space == "HSV":
            data = data[:, :, 2]
        elif self.color_space == "YCR_CB":
            data = data[:, :, 0]
        else:
            raise NotImplementedError(
                "Conversion from %s to GRAY is not ready" % self.color_space)
        sobel_xy = tuple(cv2.Sobel(data, cv2.CV_32F, *d, ksize=3)
                         for d in ((1, 0), (0, 1)))
        return numpy.linalg.norm(sobel_xy)

    def crop_image(self, data):
        crop_hw_yx = [[0, 0], [0, 0]]
        for i in 0, 1:
            crop_hw_yx[0][i] = self.crop[i] if isinstance(self.crop[i], int) \
                else int(self.crop[i] * data.shape[i])
            crop_hw_yx[1][i] = self.prng.randint(
                0, data.shape[i] - crop_hw_yx[0][i] + 1)
        crop_first = crop_hw_yx[1]
        crop_last = tuple(crop_hw_yx[1][i] + crop_hw_yx[0][i]
                          for i in (0, 1))
        return data[crop_first[0]:crop_last[0], crop_first[1]:crop_last[1]]

    def distort(self, data, mirror, rot):
        if mirror:
            data = cv2.flip(data, 1)
        data = numpy.resize(data, data.shape[:2] + (data.shape[-1] + 1,))
        data[:, :, -1] = 1
        center = tuple(reversed(tuple(data.shape[i] // 2 for i in (0, 1))))
        rot_matrix = cv2.getRotationMatrix2D(
            center, rot * 180 / numpy.pi, 1.0)
        data = cv2.warpAffine(data, rot_matrix,
                              tuple(reversed(data.shape[:2])))
        real = data[:, :, :-1]
        imag = data[:, :, -1]
        real *= imag[..., None]
        real += self.background * (1 - imag)[..., None]
        return real

    def get_distortion_by_index(self, index):
        index //= self.crop_number
        if self.mirror is True:
            return index % 2 == 1, self.rotations[index // 2]
        elif self.mirror == "random":
            mirror = bool(self.prng.randint(2))
        else:
            mirror = False
        return mirror, self.rotations[index]

    def load_keys(self, keys, pbar, data, labels, crop=True):
        """Loads data from the specified keys.
        """
        index = 0
        has_labels = False
        for key in keys:
            obj = self._load_image(key)
            label, has_labels = self._load_label(key, has_labels)
            if (self.crop is None or not crop) and \
                    obj.shape[:2] != self.uncropped_shape:
                self.debug("Ignored %s (label %s): shape %s",
                           key, label, obj.shape[:2])
                continue
            if data is not None:
                data[index] = obj
            if labels is not None:
                labels[index] = label
            index += 1
            if pbar is not None:
                pbar.inc()
        return has_labels

    def initialize(self, **kwargs):
        super(ImageLoader, self).initialize(**kwargs)
        self._restored_from_pickle = False
        if self.background_image is not None:
            self._background = self.background_image
        else:
            self._background = numpy.zeros(self.shape)
            self._background[:] = self.background_color

    def load_data(self):
        try:
            super(ImageLoader, self).load_data()
        except AttributeError:
            pass
        if self._restored_from_pickle:
            return
        for keys in self.class_keys:
            del keys[:]
        for index, class_name in enumerate(CLASS_NAME):
            keys = self.get_keys(index)
            self.class_keys[index].extend(keys)
            self.class_lengths[index] = len(keys) * self.samples_inflation
            self.class_keys[index].sort()
        if self.uncropped_shape == tuple():
            raise error.BadFormatError(
                "uncropped_shape was not initialized in get_keys()")

        # Perform a quick (unreliable) test to determine if we have labels
        keys = []
        for i in range(3):
            keys = self.class_keys[i]
            if len(keys) > 0:
                break
        assert len(keys) > 0
        data = numpy.zeros((1,) + self.shape, dtype=self.source_dtype)
        labels = numpy.zeros((1,), dtype=Loader.LABEL_DTYPE)
        self._has_labels = self.load_keys(
            (keys[numpy.random.randint(len(keys))],), None, data, labels)

    def create_minibatches(self):
        self.minibatch_data.reset()
        self.minibatch_data.mem = numpy.zeros(
            (self.max_minibatch_size,) + self.shape, dtype=self.dtype)

        self.minibatch_labels.reset()
        if self.has_labels:
            self.minibatch_labels.mem = numpy.zeros(
                (self.max_minibatch_size,), dtype=Loader.LABEL_DTYPE)

        self.minibatch_indices.reset()
        self.minibatch_indices.mem = numpy.zeros(
            self.max_minibatch_size, dtype=Loader.INDEX_DTYPE)

    def keys_from_indices(self, indices):
        keys = []
        for si in indices:
            class_index, key_remainder = self.class_index_by_sample_index(si)
            key_index = self.class_lengths[class_index] - key_remainder
            keys.append(self.class_keys[class_index][
                (key_index - 1) // self.samples_inflation])
        return keys

    def fill_minibatch(self):
        indices = self.minibatch_indices.mem[:self.minibatch_size]
        keys = self.keys_from_indices(indices)
        assert self.has_labels == self.load_keys(
            keys, None, self.minibatch_data.mem[-len(keys):],
            self.minibatch_labels.mem[-len(keys):])
        if self.samples_inflation == 1:
            return
        for index in indices:
            key_index, dist_index = divmod(index, self.samples_inflation)
            src_key_index = len(keys) - key_index - 1
            self.minibatch_data[key_index] = self.distort(
                self.minibatch_data[src_key_index],
                *self.get_distortion_by_index(dist_index))
            self.minibatch_labels[index] = self.minibatch_labels[src_key_index]

    def _load_image(self, key, crop=True):
        data = self.get_image_data(key)
        _, color = self.get_image_info(key)
        return self.preprocess_image(data, color, crop=crop)

    def _load_label(self, key, has_labels):
        label = self.get_image_label(key)
        if label is not None:
            has_labels = True
            assert isinstance(label, int), \
                "Got non-integer label %s of type %s for %s" % (
                    label, label.__class__, key)
        if has_labels and label is None:
            raise error.BadFormatError(
                "%s does not have a label, but others do" % key)
        return label, has_labels


class ImageLoaderMSEMixin(LoaderMSEMixin):
    """
    Implementation of ImageLoaderMSE for parallel inheritance.

    Attributes:
        target_keys: additional key list of targets.
    """
    def __init__(self, workflow, **kwargs):
        super(ImageLoaderMSEMixin, self).__init__(workflow, **kwargs)
        self.target_keys = []
        self.target_label_map = None

    def load_data(self):
        super(ImageLoaderMSEMixin, self).load_data()
        if self._restored_from_pickle:
            return
        self.target_keys.extend(self.get_keys(TARGET))
        length = len(self.target_keys)
        if len(set(self.target_keys)) < length:
            raise error.BadFormatError("Some targets have duplicate keys")
        self.target_keys.sort()
        if not self.has_labels and length != sum(self.class_lengths):
            raise error.BadFormatError(
                "Number of class samples %d differs from the number of "
                "targets %d" % (sum(self.class_lengths), length))
        if self.has_labels:
            labels = numpy.zeros(length, dtype=Loader.LABEL_DTYPE)
            assert self.load_keys(self.target_keys, None, None, labels)
            if len(set(labels)) < length:
                raise error.BadFormatError("Targets have duplicate labels")
            self.target_label_map = {l: self.target_keys[l] for l in labels}

    def create_minibatches(self):
        super(ImageLoaderMSEMixin, self).create_minibatches()
        self.minibatch_targets.reset()
        self.minibatch_targets.mem = numpy.zeros(
            (self.max_minibatch_size,) + self.shape, dtype=self.dtype)

    def fill_minibatch(self):
        super(ImageLoaderMSEMixin, self).fill_minibatch()
        indices = self.minibatch_indices.mem[:self.minibatch_size]
        if not self.has_labels:
            keys = self.keys_from_indices(self.shuffled_indices[i]
                                          for i in indices)
        else:
            keys = []
            for label in self.minibatch_labels.mem:
                keys.append(self.target_label_map[label])
        assert self.has_labels == self.load_keys(
            keys, None, self.minibatch_targets.mem[-len(keys):], None)
        if self.samples_inflation == 1:
            return
        # The only thing we can do is copy the same target for all distortions
        for index in indices:
            key_index, _ = divmod(index, self.samples_inflation)
            self.minibatch_targets[index] = \
                self.minibatch_targets[len(keys) - key_index - 1]


class ImageLoaderMSE(ImageLoaderMSEMixin, ImageLoader):
    """
    Loads images in MSE schemes. Like ImageLoader, mostly useful for large
    datasets.
    """
    pass


class IFileImageLoader(Interface):
    def get_label_from_filename(filename):
        """Retrieves label for the specified file path.
        """


class FileImageLoaderBase(ImageLoader):
    """
    Base class for loading something from files. Function is_valid_fiename()
    should be used in child classes as filter for loading data.
    """
    def __init__(self, workflow, **kwargs):
        super(FileImageLoaderBase, self).__init__(workflow, **kwargs)
        self._filename_types = kwargs.get("filename_types", ["jpeg"])
        self._ignored_files = kwargs.get("ignored_files", [])
        self._included_files = kwargs.get("included_files", [".*"])
        self._blacklist_regexp = re.compile(
            "^%s$" % "|".join(self.ignored_files))
        self._whitelist_regexp = re.compile(
            "^%s$" % "|".join(self.included_files))

    @property
    def filename_types(self):
        return self._filename_types

    @filename_types.setter
    def filename_types(self, value):
        del self._filename_types[:]
        if isinstance(value, str):
            self._filename_types.append(value)
        else:
            self._filename_types.extend(value)

    @property
    def ignored_files(self):
        return self._ignored_files

    @ignored_files.setter
    def ignored_files(self, value):
        del self._ignored_files[:]
        if isinstance(value, str):
            self._ignored_files.append(value)
        else:
            self._ignored_files.extend(value)

    @property
    def included_files(self):
        return self._included_files

    @included_files.setter
    def included_files(self, value):
        del self._included_files[:]
        if isinstance(value, str):
            self._included_files.append(value)
        else:
            self._included_files.extend(value)

    def get_image_info(self, key):
        """
        :param key: The full path to the analysed image.
        :return: tuple (image size, number of channels).
        """
        try:
            with open(key, "rb") as fin:
                img = Image.open(fin)
                return tuple(reversed(img.size)), MODE_COLOR_MAP[img.mode]
        except Exception as e:
            self.warning("Failed to read %s with PIL: %s", key, e)
            # Unable to read the image with PIL. Fall back to slow OpenCV
            # method which reads the whole image.
            img = cv2.imread(key, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise error.BadFormatError("Unable to read %s" % key)
            return img.shape[:2], "BGR"

    def get_image_data(self, key):
        """
        Loads data from image and normalizes it.

        Returns:
            :class:`numpy.ndarrayarray`: if there was one image in the file.
            tuple: `(data, labels)` if there were many images in the file
        """
        try:
            with open(key, "rb") as fin:
                img = Image.open(fin)
                if img.mode in ("P", "CMYK"):
                    return numpy.array(img.convert("RGB"),
                                       dtype=self.source_dtype)
                else:
                    return numpy.array(img, dtype=self.source_dtype)
        except Exception as e:
            self.warning("Failed to read %s with PIL: %s", key, e)
            img = cv2.imread(key)
            if img is None:
                raise error.BadFormatError("Unable to read %s" % key)
            return img.astype(self.source_dtype)

    def get_image_label(self, key):
        return self.get_label_from_filename(key)

    def analyze_images(self, files, pathname):
        # First pass: get the final list of files and shape
        self.debug("Analyzing %d images in %s", len(files), pathname)
        uniform_files = []
        for file in files:
            size, color_space = self.get_image_info(file)
            shape = size + (COLOR_CHANNELS_MAP[color_space],)
            if self.uncropped_shape != tuple() and \
                    shape[:2] != self.uncropped_shape:
                self.warning("%s has the different shape %s (expected %s)",
                             file, shape[:2], self.uncropped_shape)
            else:
                if self.uncropped_shape == tuple():
                    self.uncropped_shape = shape
                uniform_files.append(file)
        return uniform_files

    def is_valid_filename(self, filename):
        """Filters the file names. Return True if the specified file path must
-        be included, otherwise, False.
        """
        if self._blacklist_regexp.match(filename):
            self.debug("Ignored %s (in black list)", filename)
            return False
        if not self._whitelist_regexp.match(filename):
            self.debug("Ignored %s (not in white list)", filename)
            return False
        mime = guess_type(filename)[0]
        if mime is None:
            self.debug("Could not determine MIME type of %s", filename)
            return False
        if not mime.startswith("image/"):
            self.debug("Ignored %s (MIME is not an image)", filename)
            return False
        mime_type_name = mime[len("image/"):]
        if mime_type_name not in self.filename_types:
            self.debug("Ignored %s (MIME %s not in the list)",
                       filename, mime_type_name)
            return False
        return True


class FileListImageLoader(FileImageLoaderBase):
    """
    Input: text file, with each line giving an image filename and label
    As with ImageLoader, it is useful for large datasets.
    """
    MAPPING = "file_list_image"

    def __init__(self, workflow, **kwargs):
        super(FileListImageLoader, self).__init__(workflow, **kwargs)
        self.path_to_test_text_file = kwargs.get("path_to_test_text_file", "")
        self.path_to_val_text_file = kwargs.get("path_to_val_text_file", "")
        self.path_to_train_text_file = kwargs.get(
            "path_to_train_text_file", "")
        self.labels = {}

    def scan_files(self, pathname):
        self.info("Scanning %s..." % pathname)
        files = []
        with open(pathname, "r") as fin:
            for line in fin:
                path_to_image, _, label = line.partition(' ')
                self.labels[path_to_image] = label if label else None
                files.append(path_to_image)
        if not len(files):
            self.warning("No files were taken from %s" % pathname)
            return [], []
        return files

    def get_label_from_filename(self, filename):
        label = self.labels[filename]
        return label

    def get_keys(self, index):
        paths = (
            self.path_to_test_text_file,
            self.path_to_val_text_file,
            self.path_to_train_text_file)[index]
        if paths is None:
            return []
        return list(
            chain.from_iterable(
                self.analyze_images(self.scan_files(p), p) for p in paths))


@implementer(IImageLoader)
class FileImageLoader(FileImageLoaderBase):
    """Loads images from multiple folders. As with ImageLoader, it is useful
    for large datasets.

    Attributes:
        test_paths: list of paths with mask for test set,
                    for example: ["/tmp/\*.png"].
        validation_paths: list of paths with mask for validation set,
                          for example: ["/tmp/\*.png"].
        train_paths: list of paths with mask for train set,
                     for example: ["/tmp/\*.png"].

    Must be overriden in child class:
        get_label_from_filename()
        is_valid_filename()
    """

    def __init__(self, workflow, **kwargs):
        super(FileImageLoader, self).__init__(workflow, **kwargs)
        self.test_paths = kwargs.get("test_paths", [])
        self.validation_paths = kwargs.get("validation_paths", [])
        self.train_paths = kwargs.get("train_paths", [])
        self.verify_interface(IFileImageLoader)

    def _check_paths(self, paths):
        if not hasattr(paths, "__iter__"):
            raise TypeError("Paths must be iterable, e.g., a list or a tuple")

    @property
    def test_paths(self):
        return self._test_paths

    @test_paths.setter
    def test_paths(self, value):
        self._check_paths(value)
        self._test_paths = value

    @property
    def validation_paths(self):
        return self._validation_paths

    @validation_paths.setter
    def validation_paths(self, value):
        self._check_paths(value)
        self._validation_paths = value

    @property
    def train_paths(self):
        return self._train_paths

    @train_paths.setter
    def train_paths(self, value):
        self._check_paths(value)
        self._train_paths = value

    def scan_files(self, pathname):
        self.info("Scanning %s..." % pathname)
        files = []
        for basedir, _, filelist in os.walk(pathname):
            for name in filelist:
                full_name = os.path.join(basedir, name)
                if self.is_valid_filename(full_name):
                    files.append(full_name)
        if not len(files):
            self.warning("No files were taken from %s" % pathname)
            return [], []
        return files

    def get_keys(self, index):
        paths = (self.test_paths, self.validation_paths,
                 self.train_paths)[index]
        if paths is None:
            return []
        return list(
            chain.from_iterable(
                self.analyze_images(self.scan_files(p), p) for p in paths))


@implementer(IFileImageLoader)
class AutoLabelFileImageLoader(FileImageLoader):
    """
    FileImageLoader modification which takes labels by regular expression from
    file names. Unique selection groups are tracked and enumerated.
    """

    MAPPING = "auto_label_file_image"

    def __init__(self, workflow, **kwargs):
        super(AutoLabelFileImageLoader, self).__init__(workflow, **kwargs)
        # The default label is the parent directory
        self.label_regexp = re.compile(kwargs.get(
            "label_regexp", ".*%(sep)s([^%(sep)s]+)%(sep)s[^%(sep)s]+$" %
            {"sep": "\\" + os.sep}))
        self.unique_labels = {}
        self.labels_count = 0

    def get_label_from_filename(self, filename):
        match = self.label_regexp.search(filename)
        if match is None:
            raise error.BadFormatError(
                "%s does not match label RegExp %s" %
                (filename, self.label_regexp.pattern))
        name = match.group(1)
        if name not in self.unique_labels:
            self.unique_labels[name] = self.labels_count
            self.labels_count += 1
        return self.unique_labels[name]


class FileImageLoaderMSEMixin(ImageLoaderMSEMixin):
    """
    FileImageLoaderMSE implementation for parallel inheritance.

    Attributes:
        target_paths: list of paths for target in case of MSE.
    """

    def __init__(self, workflow, **kwargs):
        super(FileImageLoaderMSEMixin, self).__init__(
            workflow, **kwargs)
        self.target_paths = kwargs["target_paths"]

    @property
    def target_paths(self):
        return self._target_paths

    @target_paths.setter
    def target_paths(self, value):
        self._check_paths(value)
        self._target_paths = value

    def get_keys(self, index):
        if index != TARGET:
            return super(FileImageLoaderMSEMixin, self).get_keys(
                index)
        return list(chain.from_iterable(
            self.scan_files(p) for p in self.target_paths))


class FileImageLoaderMSE(FileImageLoaderMSEMixin, FileImageLoader):
    """
    MSE modification of  FileImageLoader class.
    """
    pass
