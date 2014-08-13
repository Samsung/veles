# encoding: utf-8
"""
Thic script takes a JSON file with detection results and draws pics with their
BBOXes (sequentially).
"""

import argparse
import cv2
import json
import numpy

from veles.znicz.tests.research.imagenet.forward_bbox import BBox


def create_commandline_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str,
                        help='Input JSON (should be set!)')
    return parser


if __name__ == "__main__":
    args = create_commandline_parser().parse_args()
    in_path = args.input
    in_data = json.load(open(in_path, 'r'))
    for key, val in in_data.items():
        pic_path = val["path"]
        pic = cv2.imread(pic_path)
        bboxes = val["bbxs"]
        bgr_color = numpy.random.randint(low=150, high=255, size=3).tolist()
        for bbox in bboxes:
            bbox_obj = BBox.from_json_dict(bbox)
            pic = bbox_obj.draw_on_pic(pic, line_width=2, bgr_color=bgr_color,
                                       text="%.4f" % bbox["conf"])

        cv2.imshow(key, pic)
        cv2.waitKey(0)
