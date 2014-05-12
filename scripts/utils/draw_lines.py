# encoding: utf-8
"""
Copyright (c) 2013 Samsung Electronics Co., Ltd.

This script makes a synthetic dataset with different tilt lines.
Tilt agles are are: +45, -45, 0 and +90 degrees.

.. argparse::
   :module: utils.draw_lines
   :func: create_commandline_parser
   :prog: draw_lines
"""

import argparse
from enum import IntEnum
import cv2
import numpy as np
from numpy.random import uniform
import os
import shutil


def create_commandline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filescount", type=int, default=50,
                        help='Number of files of each class')
    parser.add_argument("-l", "--linescount", type=int, default=10,
                         help="Approximate number of lines on each pic")
    parser.add_argument("-s", "--size", type=int, default=256,
                        help="Pic size")
    parser.add_argument("-n", "--noise", type=float, default=40,
                        help="Noise amplitude")
    parser.add_argument("-b", "--maxblur", type=int, default=15,
                        help="Max blur kernel size (odd values only)")
    parser.add_argument("output", type=str,
                        help='Output directory (should be set!)')
    return parser


class ImageLabel(IntEnum):
    """An enum for different tilt types: vertical, horizontal, etc."""
    vertical = 0  # |
    horizontal = 1  # --
    tilted_bottom_to_top = 2  # left lower --> right top (/)
    tilted_top_to_bottom = 3  # left top --> right bottom (\)


def draw_one_line(w, h, label, line_width, blur_stddev, clearance=0):
    """
    Creates an image with one line.

    Args:
        w (int): image width
        h (int): image height
        blur_stddev (int): gaussian blur kernel size (must be :math:`2N+1`)
        label (:class:`ImageLabel`): image tilt label
        clearance(int): minimal clearance from lines to picture corners.
    Returns:
        :class:`numpy.ndarray(dtype=uint8)`
    """

    hue = round(uniform(low=0, high=179))
    saturation = round(uniform(low=10, high=255))
    value = round(uniform(low=10, high=255))

    hsv = np.uint8([[[hue, saturation, value]]])
    [b, g, r] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]

    img = np.zeros(shape=(h, w, 3), dtype=np.uint8)

    if label == ImageLabel.vertical:
        x_start = round(uniform(low=clearance, high=(w - clearance)))
        x_stop = x_start
        y_start = 0
        y_stop = h - 1
    elif label == ImageLabel.horizontal:
        x_start = 0
        x_stop = w - 1
        y_start = round(uniform(low=clearance, high=(h - clearance)))
        y_stop = y_start
    elif label == ImageLabel.tilted_bottom_to_top:
        x_start = 0
        x_stop = w - 1
        y_start = round(uniform(low=clearance, high=(2 * h - 1 - clearance)))
        y_stop = y_start - h
    elif label == ImageLabel.tilted_top_to_bottom:
        x_start = 0
        x_stop = w - 1
        y_start = round(uniform(low=(clearance - h), high=(h - clearance)))
        y_stop = y_start + h

    cv2.line(img, (x_start, y_start), (x_stop, y_stop),
             (int(b), int(g), int(r)), line_width)
    img = cv2.GaussianBlur(img, (blur_stddev, blur_stddev), 0)
    return img


def add_images(top_img, top_alpha, bot_img, bottom_alpha):
    """
    Mixes single-line images of equal size as top and bottom ones.
    Alpha-channels are taken as:

    :math:`\\alpha'_{top}=\\alpha_{top}`

    :math:`\\alpha'_{bot}= \\min(1 - \\alpha_{top}, \\alpha_{bot})`

    Args:
        top_img (:class:`numpy.ndarray(dtype=uint8)`): top image
        bot_img (:class:`numpy.ndarray(dtype=uint8)`): bottom image
        top_alpha(:class:`numpy.ndarray(dtype=float32)`): top alpha channel
            mask, :math:`\\in [-1, 1]`.
        bot_alpha(:class:`numpy.ndarray(dtype=float32)`): bottom alpha channel
            mask, :math:`\\in [-1, 1]`.
    Returns:
        :class:`numpy.ndarray(dtype=uint8)`: mixed image
    """
    assert top_img.shape == bot_img.shape
    alpha_total = np.maximum(top_alpha, bottom_alpha)
    img_out = np.zeros(shape=top_img.shape, dtype=np.uint8)

    eff_alpha_top = top_alpha
    eff_alpha_bottom = np.minimum(bottom_alpha, 1. - top_alpha)

    for i in range(3):
        img_out[:, :, i] = eff_alpha_top * top_img[:, :, i] + \
            eff_alpha_bottom * bot_img[:, :, i]
    return img_out, alpha_total


def image_with_multi_lines(num_of_lines, type_of_label, w, h, noise_amp=0,
                           max_blur=15):
    """
    Creates image with multiple lines of the same type. Adds gaussian noise.

    Args:
        num_of_lines (int): number of lines
        type_of_label (:class:`ImageLabel`): line tilt type
        w (int): image width
        h (int): image height
        noise_amp (float): noise amplitude
        max_blur (int): max blur radius (should be odd)
    Returns:
        :class:`numpy.ndarray(dtype=uint8)`
    """

    min_line_width = 1
    max_line_width = 7

    min_smooth = 1
    max_smooth = max_blur
    assert max_smooth % 2 == 1
    assert max_smooth >= min_smooth

    clearance = int(min(w, h) / 20)  # approx. 5% from corners

    out_image = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    out_mask = np.zeros(shape=(h, w), dtype=np.uint8)

    for _ in range(num_of_lines):
        line_width = round(uniform(low=min_line_width, high=max_line_width))
        smooth_radius = round(np.random.randint(
                    low=0, high=((max_smooth - min_smooth) / 2 + 1))) * 2 + 1

        img = draw_one_line(w, h, type_of_label, line_width,
                         smooth_radius, clearance)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        alpha = gray_img.astype(dtype=np.float32) / float(np.max(gray_img))
        out_image, out_mask = add_images(img, alpha, out_image, out_mask)

    if noise_amp > 0:
        noise = np.random.normal(loc=0, scale=noise_amp, size=img.shape)
        out_image = np.clip(out_image.astype(dtype=np.float32) + noise, 0,
                            255).astype(dtype=np.uint8)

    return out_image

if __name__ == "__main__":
    parser = create_commandline_parser()
    args = parser.parse_args()

    out_dir = args.output
    learn_size = args.filescount
    test_size = args.filescount
    lines_count = args.linescount
    w, h = args.size, args.size
    noise_amp = args.noise
    max_blur = args.maxblur

    shutil.rmtree(out_dir, ignore_errors=True)

    for task_name, task_size in ("learn", learn_size), ("test", test_size):
        task_dir = os.path.join(out_dir, task_name)
        os.makedirs(task_dir)
        for label in ImageLabel:
            label_name = str(label)
            label_name = label_name[label_name.find(".") + 1:]
            label_dir = os.path.join(task_dir, label_name)
            os.makedirs(label_dir)
            for i in range(task_size):
                img = image_with_multi_lines(lines_count, label, w, h,
                                             max_blur=max_blur,
                                             noise_amp=noise_amp)
                cv2.imwrite(
                    os.path.join(label_dir, "%s_%s_%i.jpg" %
                                  (task_name, label_name, i)), img)
