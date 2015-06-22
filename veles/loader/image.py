# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Aug 14, 2013

Ontology of image loading classes.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


from __future__ import division
from collections import defaultdict
from itertools import chain
try:
    import cv2
except ImportError:
    pass
import numpy
from PIL import Image
from zope.interface import implementer, Interface

from veles.compat import from_none
import veles.error as error
from veles.external.progressbar import ProgressBar
from veles.loader.base import CLASS_NAME, ILoader, Loader, \
    TRAIN, VALID, TEST, LoaderError, LoaderWithValidationRatio
from veles.memory import Array
from veles.prng import RandomGenerator


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
class ImageLoader(LoaderWithValidationRatio):
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
        self._source_dtype = numpy.float32
        self._original_shape = tuple()
        self.class_keys = [[], [], []]
        self.verify_interface(IImageLoader)
        self.path_to_mean = kwargs.get("path_to_mean", None)
        self.add_sobel = kwargs.get("add_sobel", False)
        self.mirror = kwargs.get("mirror", False)  # True, False, "random"
        self.scale = kwargs.get("scale", 1.0)
        self.scale_maintain_aspect_ratio = kwargs.get(
            "scale_maintain_aspect_ratio", True)
        self.rotations = kwargs.get("rotations", (0.0,))  # radians
        self.crop = kwargs.get("crop", None)
        self.crop_number = kwargs.get("crop_number", 1)
        self._background = None
        self.background_image = kwargs.get("background_image", None)
        self.background_color = kwargs.get(
            "background_color", (0xff, 0x14, 0x93))
        self.smart_crop = kwargs.get("smart_crop", True)
        self.minibatch_label_values = Array()

    @property
    def source_dtype(self):
        return self._source_dtype

    @property
    def color_space(self):
        return self._color_space

    @color_space.setter
    def color_space(self, value):
        self._validate_color_space(value)
        self._color_space = value

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
        if not isinstance(self.scale, tuple):
            if self._original_shape == tuple():
                return tuple()
            return self._scale_shape(self._original_shape)[:2]
        else:
            return self.scale

    @property
    def original_shape(self):
        return self._original_shape

    @original_shape.setter
    def original_shape(self, value):
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
        self._original_shape = value

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
    def smart_crop(self):
        """
        :return: Value indicating whether to crop only around bboxes.
        """
        return self._smart_crop

    @smart_crop.setter
    def smart_crop(self, value):
        if not isinstance(value, bool):
            raise TypeError("smart_crop must be a boolean value")
        self._smart_crop = value

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
            if getattr(self, "background_color", None) is not None:
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
        if getattr(self, "background_image", None) is not None:
            self.warning(
                "background_color = %s is ignored in favor of "
                "background_image", value)
        self._background_color = value

    @property
    def background(self):
        if self._background is None:
            if self.background_image is not None:
                self._background = self.background_image
            else:
                self._background = numpy.zeros(self.shape)
                self._background[:] = self.background_color
        return self._background.copy()

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
            return self._scale_shape(info[0]), info[1]

    def get_image_bbox(self, key, size):
        """
        Override this method for custom label <-> bbox mapping.
        :param key: The image key.
        :param size: The image size (for optimization purposes).
        :return: (ymin, ymax, xmin, xmax).
        """
        return 0, size[0], 0, size[1]

    def preprocess_image(self, data, color, crop, bbox):
        """
        Transforms images before serving.
        :param data: the loaded image data.
        :param color: The loaded image color space.
        :param crop: True if must crop the scaled image; otherwise, False.
        :param bbox: The bounding box of the labeled object. Tuple
        (ymin, ymax, xmin, xmax).
        :return: The transformed image data, the label value (from 0 to 1).
        """
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
            data, bbox = self.scale_image(data, bbox)
        if crop and self.crop is not None:
            data, label_value = self.crop_image(data, bbox)
        else:
            label_value = 1

        return data, label_value, bbox

    def scale_image(self, data, bbox):
        bbox = numpy.array(bbox, float)
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
            data = cv2.resize(
                data, (dst_width, dst_height),
                interpolation=cv2.INTER_CUBIC)
            dst_x_max = dst_x_min + data.shape[1]
            dst_y_max = dst_y_min + data.shape[0]
            sample = self.background
            sample[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = data
            data = sample.copy()
            bbox[:2] *= (dst_y_max - dst_y_min) / (bbox[1] - bbox[0])
            bbox[:2] += dst_y_min
            bbox[2:] *= (dst_x_max - dst_x_min) / (bbox[3] - bbox[2])
            bbox[2:] += dst_x_min
        else:
            data = cv2.resize(
                data, tuple(reversed(self.uncropped_shape[:2])),
                interpolation=cv2.INTER_CUBIC)
            bbox[:2] *= self.uncropped_shape[0] / (bbox[1] - bbox[0])
            bbox[2:] *= self.uncropped_shape[1] / (bbox[3] - bbox[2])
        return data, tuple(bbox.astype(numpy.int32))

    def add_sobel_channel(self, data):
        original_data = data
        if self.channels_number == 1 + 1:
            original_data = original_data.reshape(
                original_data.shape[:2] + (1,))
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
        sobel_data = numpy.zeros(
            shape=data.shape + (original_data.shape[2] + 1,),
            dtype=original_data.dtype)
        sobel_data[:, :, -1] = numpy.linalg.norm(sobel_xy)
        sobel_data[:, :, :-1] = original_data
        return sobel_data

    def crop_image(self, data, bbox):
        """
        Cuts a rectangular part of an image.
        :param data: The source image to crop.
        :param bbox: (ymin, ymax, xmin, xmax)
        :return: tuple (image part randomly cropped around the bbox,\
        intersection ratio)
        """
        crop_hw_yx = [[0, 0], [0, 0]]
        for i in 0, 1:
            crop_hw_yx[0][i] = self.crop[i] if isinstance(self.crop[i], int) \
                else int(self.crop[i] * data.shape[i])
            crop_size = crop_hw_yx[0][i]
            crop_hw_yx[1][i] = self.prng.randint(
                max(bbox[i * 2] - crop_size, 0),
                min(data.shape[i] - crop_size + 1,
                    bbox[i * 2 + 1] + crop_size))
        crop_first = crop_hw_yx[1]
        crop_last = tuple(crop_hw_yx[1][i] + crop_hw_yx[0][i]
                          for i in (0, 1))
        crop_bbox = crop_first[0], crop_last[0], crop_first[1], crop_last[1]
        return data[crop_bbox[0]:crop_bbox[1], crop_bbox[2]:crop_bbox[3]], \
            self._intersection(bbox, crop_bbox)

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

    def load_keys(self, keys, pbar, data, labels, label_values, crop=True):
        """Loads data from the specified keys.
        """
        index = 0
        has_labels = False
        for key in keys:
            obj, label_value, _ = self._load_image(key)
            label, has_labels = self._load_label(key, has_labels)
            if (self.crop is None or not crop) and \
                    obj.shape[:2] != self.uncropped_shape:
                self.warning(
                    "Ignored %s (label %s): shape %s",
                    key, label, obj.shape[:2])
                continue
            if data is not None:
                data[index] = obj
            if labels is not None:
                labels[index] = label
            if label_values is not None:
                label_values[index] = label_value
            index += 1
            if pbar is not None:
                pbar.inc()
        return has_labels

    def load_labels(self):
        if not self.has_labels:
            return
        self.info("Reading labels...")
        different_labels = defaultdict(int), defaultdict(int), defaultdict(int)
        label_key_map = defaultdict(list), defaultdict(list), defaultdict(list)
        pb = ProgressBar(maxval=self.total_samples, term_width=40)
        pb.start()
        for class_index in range(3):
            for key in self.class_keys[class_index]:
                label, has_labels = self._load_label(key, True)
                assert has_labels
                different_labels[class_index][label] += 1
                label_key_map[class_index][label].append(key)
                self._samples_mapping[label].add(key)
                pb.inc()
        pb.finish()

        return different_labels, label_key_map

    def initialize(self, **kwargs):
        self._restored_from_pickle_ = kwargs["snapshot"]
        super(ImageLoader, self).initialize(**kwargs)
        del self._restored_from_pickle_

    def load_data(self):
        try:
            super(ImageLoader, self).load_data()
        except AttributeError:
            pass
        if self._restored_from_pickle_:
            self.info("Scanning for changes...")
            progress = ProgressBar(maxval=self.total_samples, term_width=40)
            progress.start()
            for keys in self.class_keys:
                for key in keys:
                    progress.inc()
                    size, _ = self.get_effective_image_info(key)
                    if size != self.uncropped_shape:
                        raise error.BadFormatError(
                            "%s changed the effective size (now %s, was %s)" %
                            (key, size, self.uncropped_shape))
            progress.finish()
            return
        for keys in self.class_keys:
            del keys[:]
        for index, class_name in enumerate(CLASS_NAME):
            keys = set(self.get_keys(index))
            self.class_keys[index].extend(keys)
            self.class_lengths[index] = len(keys) * self.samples_inflation
            self.class_keys[index].sort()

        if self.uncropped_shape == tuple():
            raise error.BadFormatError(
                "original_shape was not initialized in get_keys()")
        self.info(
            "Found %d samples of shape %s (%d TEST, %d VALIDATION, %d TRAIN)",
            self.total_samples, self.shape, *self.class_lengths)

        # Perform a quick (unreliable) test to determine if we have labels
        keys = next(k for k in self.class_keys if len(k) > 0)
        self._has_labels = self.load_keys(
            (keys[RandomGenerator(None).randint(len(keys))],),
            None, None, None, None)
        self._resize_validation_keys(self.load_labels())

    def create_minibatch_data(self):
        self.minibatch_data.reset(numpy.zeros(
            (self.max_minibatch_size,) + self.shape, dtype=self.dtype))

        self.minibatch_label_values.reset(numpy.zeros(
            self.max_minibatch_size, numpy.float32))

    def keys_from_indices(self, indices):
        for index in indices:
            class_index, origin_index, _ = \
                self._get_class_origin_distortion_from_index(index)
            yield self.class_keys[class_index][origin_index]

    def fill_minibatch(self):
        indices = self.minibatch_indices.mem[:self.minibatch_size]
        assert self.has_labels == self.load_keys(
            self.keys_from_indices(indices), None, self.minibatch_data.mem,
            self.raw_minibatch_labels, self.minibatch_label_values)
        if self.samples_inflation == 1:
            return
        for pos, index in enumerate(indices):
            _, _, dist_index = \
                self._get_class_origin_distortion_from_index(index)
            self.minibatch_data[pos] = self.distort(
                self.minibatch_data[pos],
                *self.get_distortion_by_index(dist_index))

    def _resize_validation_keys(self, label_analysis):
        if label_analysis is None:
            return
        different_labels, label_key_map = label_analysis
        if self.validation_ratio is None:
            self._setup_labels_mapping(different_labels)
            return
        if self.validation_ratio < 0:
            self.class_keys[TRAIN] += self.class_keys[VALID]
            self.class_lengths[TRAIN] += self.class_lengths[VALID]
            del self.class_keys[VALID][:]
            self.class_lengths[VALID] = 0
            merged = {k: (different_labels[VALID][k] +
                          different_labels)[TRAIN][k]
                      for k in label_key_map[TRAIN]}
            self._setup_labels_mapping((different_labels[TEST], {}, merged))
            return

        overall = sum(len(ck) for ck in self.class_keys[VALID:])
        target_validation_length = int(overall * self.validation_ratio)

        if not self.has_labels:
            keys = list(chain.from_iterable(self.class_keys[VALID:]))
            keys.sort()
            self.prng.shuffle(keys)
            del self.class_keys[VALID][:]
            self.class_keys[VALID].extend(keys[:target_validation_length])
            del self.class_keys[TRAIN][:]
            self.class_keys[TRAIN].extend(keys[target_validation_length:])
            self._finalize_resizing_validation(different_labels, label_key_map)
            return

        # We must ensure that each set has the same labels
        # The first step is to pick two keys for each label and distribute them
        # into VALID and TRAIN evenly
        if len(label_key_map[TRAIN]) > target_validation_length:
            raise LoaderError(
                "Unable to set the new size of the validation set to %d (%.3f)"
                " since the number of labels is %d" %
                (target_validation_length * self.samples_inflation,
                 self.validation_ratio, len(label_key_map[TRAIN])))
        if overall - target_validation_length < len(label_key_map[TRAIN]):
            raise LoaderError(
                "Unable to set the new size of the training set to %d (%.3f) "
                "since the number of labels is %d" %
                ((overall - target_validation_length) * self.samples_inflation,
                 1.0 - self.validation_ratio, len(label_key_map[TRAIN])))
        vt_label_key_map = {l: (label_key_map[VALID].get(l, []) +
                                label_key_map[TRAIN].get(l, []))
                            for l in label_key_map[TRAIN]}
        for i in VALID, TRAIN:
            del self.class_keys[i][:]
        for label, keys in sorted(vt_label_key_map.items()):
            if len(keys) < 2:
                raise LoaderError("Label %s has less than 2 keys" % label)
            choice = self.prng.choice(len(keys), 2, replace=False)
            assert choice[0] != choice[1]
            for i in VALID, TRAIN:
                self.class_keys[i].append(keys[choice[i - 1]])
            for c in sorted(choice, reverse=True):
                del keys[c]

        # Distribute the left keys randomly
        left_keys = list(sorted(chain.from_iterable(
            vt_label_key_map.values())))
        self.prng.shuffle(left_keys)
        offset_val_length = \
            target_validation_length - len(vt_label_key_map)
        self.class_keys[VALID].extend(left_keys[:offset_val_length])
        self.class_keys[TRAIN].extend(left_keys[offset_val_length:])
        self._finalize_resizing_validation(different_labels, label_key_map)

    def _finalize_resizing_validation(self, different_labels, label_key_map):
        for ck in self.class_keys[VALID:]:
            ck.sort()
        for i in VALID, TRAIN:
            self.class_lengths[i] = len(self.class_keys[i]) * \
                self.samples_inflation
        new_diff = defaultdict(int), defaultdict(int)
        key_label_map = {}
        for ci in VALID, TRAIN:
            key_label_map.update({k: l
                                  for l, keys in label_key_map[ci].items()
                                  for k in keys})
        for ci in VALID, TRAIN:
            for key in self.class_keys[ci]:
                new_diff[ci - 1][key_label_map[key]] += 1
        self._setup_labels_mapping((different_labels[TEST],) + new_diff)

    def _get_class_origin_distortion_from_index(self, index):
        class_index, key_remainder = self.class_index_by_sample_index(index)
        key_index = self.class_lengths[class_index] - key_remainder
        return (class_index,) + divmod(key_index, self.samples_inflation)

    def _load_image(self, key, crop=True):
        """Returns the data to serve corresponding to the given image key and
        the label value (from 0 to 1).
        """
        data = self.get_image_data(key)
        size, color = self.get_image_info(key)
        bbox = self.get_image_bbox(key, size)
        return self.preprocess_image(data, color, crop, bbox)

    def _load_label(self, key, has_labels):
        label = self.get_image_label(key)
        if label is not None:
            has_labels = True
        if has_labels and label is None:
            raise error.BadFormatError(
                "%s does not have a label, but others do" % key)
        return label, has_labels

    def _intersection(self, bbox_a, bbox_b):
        ymin_a, ymax_a, xmin_a, xmax_a = bbox_a
        ymin_b, ymax_b, xmin_b, xmax_b = bbox_b

        x_intersection = min(xmax_a, xmax_b) - max(xmin_a, xmin_b)
        y_intersection = min(ymax_a, ymax_b) - max(ymin_a, ymin_b)

        if int(x_intersection) | int(y_intersection) <= 0:
            return 0
        else:
            return x_intersection * y_intersection

    def _scale_shape(self, shape):
        return tuple(int(shape[i] * self.scale) for i in (0, 1)) + shape[2:]

    def _validate_color_space(self, value):
        if not isinstance(value, str):
            raise TypeError(
                "db_colorpsace must be a string (got %s)" % type(value))
        if value != "RGB" and not hasattr(cv2, "COLOR_%s2RGB" % value):
            raise ValueError("Unsupported color space: %s" % value)
