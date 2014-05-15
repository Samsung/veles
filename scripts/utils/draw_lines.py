# encoding: utf-8
"""
Copyright (c) 2013 Samsung Electronics Co., Ltd.

This script makes a synthetic dataset with different simple figures

.. argparse::
   :module: utils.draw_lines
   :func: create_commandline_parser
   :prog: draw_lines
"""

import argparse
import cv2
from enum import IntEnum
import logging
from math import sin, cos, pi
import numpy as np
from numpy.random import uniform, randint
import os
import shutil


def create_commandline_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filescount", type=int, default=50,
                        help='Number of files of each class')
    parser.add_argument("-s", "--size", type=int, default=256,
                        help="Pic size")
    parser.add_argument("-n", "--noise", type=float, default=40,
                        help="Noise amplitude")
    parser.add_argument("-b", "--maxblur", type=int, default=15,
                        help="Max blur kernel size (odd values only)")
    parser.add_argument("-i", "--inverted", action="store_true",
                        help="Invert colors on generated images")
    parser.add_argument("output", type=str,
                        help='Output directory (should be set!)')
    return parser


class ImageLabel(IntEnum):
    """An enum for different figure primitive classes"""
    vertical = 0  # |
    horizontal = 1  # --
    tilted_bottom_to_top = 2  # left lower --> right top (/)
    tilted_top_to_bottom = 3  # left top --> right bottom (\)
    straight_grid = 4  # 0 and 90 deg lines simultaneously
    tilted_grid = 5  # +45 and -45 deg lines simultaneously
    circle = 6
    square = 7
    right_angle = 8
    triangle = 9
    sinusoid = 10


def pick_color():
    """Randomly selects a BGR color.

    Returns:
        :class:`numpy.ndarray(dtype=unit8, shape=(3,))`
    """
    hue = randint(low=0, high=179)
    saturation = randint(low=10, high=255)
    value = randint(low=10, high=255)

    hsv = np.uint8([[[hue, saturation, value]]])
    [b, g, r] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return np.array([b, g, r])


def draw_line(img, start_point, stop_point, color, thickness):
    [b, g, r] = [int(x) for x in color]
    start_point = int(start_point[0]), int(start_point[1])
    stop_point = int(stop_point[0]), int(stop_point[1])
    cv2.line(img, start_point, stop_point, (int(b), int(g), int(r)), thickness)


def draw_vertical_lines(img, thickness, lines_count, clearance, bgr_color):
    h, w, _ = img.shape
    for _ in range(lines_count):
        x_start = randint(low=clearance, high=(w - clearance))
        x_stop = x_start
        y_start = 0
        y_stop = h - 1
        draw_line(img, (x_start, y_start), (x_stop, y_stop),
                 bgr_color, thickness)


def draw_horizontal_lines(img, thickness, lines_count, clearance, bgr_color):
    """
    Draws randomly located horizontal line on given image

    Args:
        img (:class:`numpy.ndarray(dtype=uint8)`) image to draw on
        clearance(int): minimal clearance from lines to picture corners
        bgr_color(array): BGR color of line
    """
    h, w, _ = img.shape
    for _ in range(lines_count):
        x_start = 0
        x_stop = w - 1
        y_start = round(uniform(low=clearance, high=(h - clearance)))
        y_stop = y_start
        draw_line(img, (x_start, y_start), (x_stop, y_stop),
                 bgr_color, thickness)


def draw_tilted_bottom_top_lines(img, thickness, lines_count,
                                 clearance, bgr_color):
    """
    Draws randomly located +45 deg tilted line on given image

    Args:
        img (:class:`numpy.ndarray(dtype=uint8)`) image to draw on
        clearance(int): minimal clearance from lines to picture corners
        bgr_color(array): BGR color of line
    """
    h, w, _ = img.shape
    for _ in range(lines_count):
        x_start = 0
        x_stop = w - 1
        y_start = round(uniform(low=clearance, high=(2 * h - 1 - clearance)))
        y_stop = y_start - h
        draw_line(img, (x_start, y_start), (x_stop, y_stop),
                 bgr_color, thickness)


def draw_tilted_top_bottom_lines(img, thickness, lines_count,
                                clearance, bgr_color):
    """
    Draws randomly located -45 deg tilted line on given image

    Args:
        img (:class:`numpy.ndarray(dtype=uint8)`) image to draw on
        clearance(int): minimal clearance from lines to picture corners
        bgr_color(array): BGR color of line
        lines_count(int): number of lines
    """
    h, w, _ = img.shape
    for _ in range(lines_count):
        x_start = 0
        x_stop = w - 1
        y_start = round(uniform(low=(clearance - h), high=(h - clearance)))
        y_stop = y_start + h
        draw_line(img, (x_start, y_start), (x_stop, y_stop),
                 bgr_color, thickness)


def draw_straight_uniform_grid(img, thickness, step, bgr_color):
    """Draws square-cell uniform grid with random shift"""
    h, w, _ = img.shape
    x, y = uniform(low=0, high=step, size=2)
    while x < w:
        draw_line(img, (x, 0), (x, h), bgr_color, thickness)
        x += step
    while y < h:
        draw_line(img, (0, y), (w, y), bgr_color, thickness)
        y += step


def draw_tilted_uniform_grid(img, thickness, step, bgr_color):
    """Draws square-cell uniform grid with random shift, tilted on 45 degs"""
    h, w, _ = img.shape
    x, y = uniform(low=-w, high=-w + step, size=2)
    while x < w:
        draw_line(img, (x, h - 1), (x + w, 0), bgr_color, thickness)
        x += step
    while y < h:
        draw_line(img, (0, y), (w, y + h - 1), bgr_color, thickness)
        y += step


def draw_circle(img, thickness, radius, clearance, bgr_color):
    """Draws a circle"""
    [b, g, r] = [int(x) for x in bgr_color]
    h, w, _ = img.shape
    x_center = randint(low=clearance + radius,
                       high=(w - 1) - clearance - radius)
    y_center = randint(low=clearance + radius,
                       high=(h - 1) - clearance - radius)
    cv2.circle(img, (x_center, y_center), radius, [b, g, r], thickness)


def draw_angle(img, thickness, radius, angle, clearance, bgr_color):
    """Draws an angle"""
    [b, g, r] = [int(x) for x in bgr_color]
    h, w, _ = img.shape

    x_center = randint(low=clearance, high=(w - 1) - clearance)
    y_center = randint(low=0 + clearance, high=(h - 1) - clearance)

    start_angle = uniform(low=0, high=2 * np.pi)
    stop_angle = start_angle + angle

    x_start = int(x_center + radius * np.cos(start_angle))
    y_start = int(y_center + radius * np.sin(start_angle))

    x_stop = int(x_center + radius * np.cos(stop_angle))
    y_stop = int(y_center + radius * np.sin(stop_angle))

    pts = np.array([[x_start, y_start], [x_center, y_center],
                    [x_stop, y_stop]], np.int32)

    cv2.polylines(img, [pts], False, [b, g, r], thickness=thickness)


def draw_triangle(img, thickness, radius, clearance, bgr_color):
    """Draws a triangle"""
    [b, g, r] = [int(x) for x in bgr_color]
    h, w, _ = img.shape

    x_center = randint(low=clearance + radius,
                       high=(w - 1) - clearance - radius)
    y_center = randint(low=0 + clearance + radius,
                       high=(h - 1) - clearance - radius)

    start_angle = uniform(low=0, high=2 * np.pi)
    points = []
    for i in range(3):
        angle = np.pi * 2. / 3. * i
        x = np.cos(angle)
        y = np.sin(angle)
        points.append([x, y])

    points = warped_polyline(points, [x_center, y_center], start_angle, 40)

    points = np.array(points, np.int32)

    cv2.polylines(img, [points], True, [b, g, r], thickness=thickness)


def draw_square(img, thickness, radius, clearance, bgr_color):
    """Draws a square"""
    [b, g, r] = [int(x) for x in bgr_color]
    h, w, _ = img.shape

    x_center = randint(low=clearance + radius,
                       high=(w - 1) - clearance - radius)
    y_center = randint(low=0 + clearance + radius,
                       high=(h - 1) - clearance - radius)

    start_angle = uniform(low=0, high=2 * np.pi)
    points = []
    for i in range(4):
        angle = start_angle + np.pi * 2. / 4. * i
        x = int(x_center + radius * np.cos(angle))
        y = int(y_center + radius * np.sin(angle))
        points.append([x, y])

    points = np.array(points, np.int32)

    cv2.polylines(img, [points], True, [b, g, r], thickness=thickness)


def draw_sinusoid(img, thickness, radius, clearance, bgr_color):
    """
    Draws a sinusoid primitive

    Args:
        img(:class:`numpy.ndarray(dtype=unit8)`): image to draw on
            (OpenCV format)
        thickness(int): line thickness
        radius(float): approx size of the primitive
        clearance(float): approx clearance from pic borders
    """
    [b, g, r] = [int(x) for x in bgr_color]
    h, w, _ = img.shape

    x_center = randint(low=clearance, high=(w - 1) - clearance)
    y_center = randint(low=0 + clearance, high=(h - 1) - clearance)

    start_angle = uniform(low=0, high=2 * np.pi)

    x_points = np.linspace(-pi / 2, pi / 2, 100)
    points = [[x / (pi / 2), sin(4 * x)] for x in x_points]

    points = warped_polyline(points, [x_center, y_center], start_angle, 40)
    points = np.array(points, np.int32)

    cv2.polylines(img, [points], False, [b, g, r], thickness=thickness)


def warped_polyline(points, pos, rot, scale):
    """
    Warps a polyline: adds position, rotation and scale

    Args:
        points(array-like): original polyline data
        pos(tuple): position (x, y)
        rot(float): rotation angle
        scale(float): scale ratio
    Returns:
        :class:`numpy.ndarray`
    """
    new_points = []
    for x, y in points:
        x_new, y_new = x * cos(rot) - y * sin(rot), x * sin(rot) + y * cos(rot)
        x_new = x_new * scale + pos[0]
        y_new = y_new * scale + pos[1]
        new_points.append([x_new, y_new])
    return np.array(new_points)


class ImageGenerator(object):
    """
    Synthetical images generator

    Args:
            max_blur(int): max blur kernel size
            noise_amp(float): additive gaussian noise amplitude
            invert(bool): invert colors on generated images
    """
    def __init__(self, max_blur, noise_amp, invert):
        self._max_blur = max_blur
        self._noise_amp = noise_amp
        self._invert = invert

    def generate(self, size, label):
        """
        Generates a square image with given params. Adds blur and noise.

        Args:
            size(int): side size of image
            label(:class:`ImageLabel`): image label
        Returns:
            :class:`numpy.ndarray(dtype=uint8)`: BGR image in OpenCV format
        """

        min_line_width = 1
        max_line_width = 7

        min_smooth = 1
        max_smooth = self._max_blur
        assert max_smooth % 2 == 1
        assert max_smooth >= min_smooth

        w, h = size, size

        clearance = min(w, h) // 20  # approx. 5% from corners

        n_lines = 10
        color = pick_color()

        img = np.zeros(shape=(h, w, 3), dtype=np.uint8)

        thickness = randint(low=min_line_width, high=max_line_width)
        blur_radius = randint(
            low=0, high=((max_smooth - min_smooth) // 2 + 1)) * 2 + 1

        fig_size = randint(low=size // 10, high=size // 3 - clearance)

        #Creating initial image
        if label == ImageLabel.vertical:
            draw_vertical_lines(img, thickness, n_lines, clearance, color)
        elif label == ImageLabel.horizontal:
            draw_horizontal_lines(img, thickness, n_lines, clearance, color)
        elif label == ImageLabel.tilted_bottom_to_top:
            draw_tilted_bottom_top_lines(img, thickness, n_lines, clearance,
                                         color)
        elif label == ImageLabel.tilted_top_to_bottom:
            draw_tilted_top_bottom_lines(img, thickness, n_lines, clearance,
                                         color)
        elif label == ImageLabel.straight_grid:
            draw_straight_uniform_grid(img, thickness, fig_size, color)
        elif label == ImageLabel.tilted_grid:
            draw_tilted_uniform_grid(img, thickness, fig_size, color)
        elif label == ImageLabel.circle:
            draw_circle(img, thickness, fig_size, clearance, color)
        elif label == ImageLabel.square:
            draw_square(img, thickness, fig_size, clearance, color)
        elif label == ImageLabel.right_angle:
            draw_angle(img, thickness, fig_size, pi / 2, clearance, color)
        elif label == ImageLabel.triangle:
            draw_triangle(img, thickness, fig_size, clearance, color)
        elif label == ImageLabel.sinusoid:
            draw_sinusoid(img, thickness, fig_size, clearance, color)

        #BLUR
        img = cv2.GaussianBlur(img, (blur_radius, blur_radius), 0)

        #NOISE
        if self._noise_amp > 0:
            noise = np.random.normal(loc=0, scale=self._noise_amp,
                                     size=img.shape)
            img = np.clip(img.astype(dtype=np.float32) + noise, 0,
                                255).astype(dtype=np.uint8)

        #INVERT
        if self._invert:
            img = 255 - img
        return img

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = create_commandline_parser()
    args = parser.parse_args()

    out_dir = args.output
    learn_size = args.filescount
    test_size = args.filescount

    generator = ImageGenerator(args.maxblur, args.noise, args.inverted)
    size = args.size

    shutil.rmtree(out_dir, ignore_errors=True)

    for task_name, task_size in ("learn", learn_size), ("test", test_size):
        task_dir = os.path.join(out_dir, task_name)
        os.makedirs(task_dir)
        for label in ImageLabel:
            logging.info(label)
            label_name = str(label)
            label_name = label_name[label_name.find(".") + 1:]
            label_dir = os.path.join(task_dir, label_name)
            os.makedirs(label_dir)
            for i in range(task_size):

                img = generator.generate(size, label)
                cv2.imwrite(
                    os.path.join(label_dir, "%s_%s_%i.jpg" %
                                (task_name, label_name, i)), img)
