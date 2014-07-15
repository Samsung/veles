# encoding: utf-8

"""
This script takes pics from ImageNet IMG dataset and tries to adjust the
BBOXes of the objects on them. Output data is put into a Pickle file.

.. argparse::
   :module: utils.bbox_adjuster
   :func: create_parser
   :prog: bbox_adjuster
"""

import argparse
import json
import numpy as np
import os
import pickle
import time

import caffe


def nms_detections(dets, overlap=0.7):
    """
    Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously
    selected detection.

    This version is translated from Matlab code by Tomasz Malisiewicz,
    who sped up Pedro Felzenszwalb's code.

    Args:
        dets(ndarray): each row is ['xmin', 'ymin', 'xmax', 'ymax', 'score']
        overlap(float): minimum overlap ratio (0.5 default)

    Returns:
        dets(ndarray): remaining after suppression.
    """
    if np.shape(dets)[0] < 1:
        return dets

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    w = x2 - x1
    h = y2 - y1
    area = w * h

    s = dets[:, 4]
    ind = np.argsort(s)

    pick = []
    counter = 0
    while len(ind) > 0:
        last = len(ind) - 1
        i = ind[last]
        pick.append(i)
        counter += 1

        xx1 = np.maximum(x1[i], x1[ind[:last]])
        yy1 = np.maximum(y1[i], y1[ind[:last]])
        xx2 = np.minimum(x2[i], x2[ind[:last]])
        yy2 = np.minimum(y2[i], y2[ind[:last]])

        w = np.maximum(0., xx2 - xx1 + 1)
        h = np.maximum(0., yy2 - yy1 + 1)

        o = w * h / area[ind[:last]]

        to_delete = np.concatenate(
            (np.nonzero(o > overlap)[0], np.array([last])))
        ind = np.delete(ind, to_delete)

    return dets[pick, :]


def load_synsets(synsets_path):
    """
    Loads synsets from `synsets_path`.

    Returns:
        synsets(:class:`list`):
        synset_names(:class:`list`):
        synset_indexes(:class:`dict`):
    """
    synsets = []
    synset_names = []
    synset_indexes = {}
    for i, line in enumerate(open(synsets_path, 'r').readlines()):
        line = line.replace("\n", "")
        synset_id = line.split(" ")[0]
        synset_name = line[len(synset_id) + 1:]
        synsets.append(synset_id)
        synset_names.append(synset_name)
        synset_indexes[synset_id] = i
    return synsets, synset_names, synset_indexes


def create_parser():
    """
    Creates a commandline parser.
    """
    parser = argparse.ArgumentParser(
        description='This script takes some images from Imagenet dataset, \
        finds their BBOXes, then saves their BBOXes to a JSON file.')

    parser.add_argument("-c", "--caffe", required=True,
                        type=str, help="path to CAFFE dir")
    parser.add_argument("-i", "--intermediate", required=False, default="",
                        type=str, help="path to dump raw bboxes")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="output path")
    parser.add_argument("input", nargs="+", help="input folders")

    return parser

#########################################################################
if __name__ == "__main__":

    args = create_parser().parse_args()
    synsets_path = os.path.join(args.caffe, "data/ilsvrc12/synset_words.txt")
    synsets, synset_names, synset_indexes = load_synsets(synsets_path)

    coord_cols = ['ymin', 'xmin', 'ymax', 'xmax']
    coord_col_ids = {}
    for i, col in enumerate(coord_cols):
        coord_col_ids[col] = i

    print("Scanning folders...")
    file_paths = []
    for root, dirs, files in os.walk(args.input[0]):
        for file in files:
            file_paths.append(os.path.join(root, file))
        print("Files found: %i,\t current dir: %s" % (len(file_paths), root))

    #Extracting raw data
    model_def = os.path.join(args.caffe,
                             "examples/imagenet/imagenet_deploy.prototxt")
    pretrained_model = os.path.join(
        args.caffe, "examples/imagenet/caffe_reference_imagenet_model")
    mean_file = os.path.join(args.caffe,
                             "python/caffe/imagenet/ilsvrc_2012_mean.npy")
    channel_swap = (2, 1, 0)

    detector = caffe.Detector(model_def, pretrained_model,
                              gpu=True, mean_file=mean_file,
                              input_scale=255, channel_swap=channel_swap)

    caffe_portion = 1  # How many pics to give to CAFFE at once

    # Calculate raw BBOXes
    results_for_files = {}
    for i in range(0, len(file_paths), caffe_portion):
        paths_to_detect = file_paths[i:i + caffe_portion]
        start_time = time.time()
        raw_results = detector.detect_selective_search(paths_to_detect)
        time_eaten = time.time() - start_time
        print("Time eaten: %.3f (%.3f per pic), pics left: %i" % (
            time_eaten, time_eaten / len(paths_to_detect),
            len(file_paths) - i - len(paths_to_detect)))
        for line in raw_results:
            fname = line["filename"]
            if not (fname in results_for_files):
                results_for_files[fname] = []
            this_file_results = results_for_files[fname]
            synset_id, file_id = fname.split("/")[-1].split(".")[0].split("_")
            window = line["window"]
            det = [int(window[coord_col_ids["xmin"]]),
                   int(window[coord_col_ids["ymin"]]),
                   int(window[coord_col_ids["xmax"]]),
                   int(window[coord_col_ids["ymax"]])]
            det += [float(line["prediction"][synset_indexes[synset_id]])]
            this_file_results.append(det)

    # Dump raw BBOXes
    if args.intermediate:
        pickle.dump(results_for_files, open(args.intermediate, 'wb'))

    # Calculate fine BBOXes and save then into a JSON
    # [{"filename": "...", windows: [xmin, xmax, ymin, ymax, score]},
    #  {"filename": "...", windows: [...]}]

    final_results = []
    for fname, dets in results_for_files.iteritems():
        fname_short = fname.split("/")[-1]
        raw_results = nms_detections(np.asarray(dets))
        result_dict = {"filename": fname_short, "windows": []}
        window_list = result_dict["windows"]
        for i in range(raw_results.shape[0]):
            one_result = raw_results[i, :]
            one_window = one_result[:-1].astype(np.int32).tolist()
            one_window.append(float(one_result[-1]))
            window_list.append(one_window)
        final_results.append(result_dict)

    # Save fine BBOXes
    json.dump(final_results, open(args.output, 'wb'), sort_keys=True, indent=4)
