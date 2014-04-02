#!/usr/bin/python3.3 -O
"""
Created on Sep 2, 2013

File for korean channels recognition.

@author: Kazantsev Alexey <a.kazantsev@samsung.com>
"""


import glymur
import logging
import numpy
import os
import pickle
import re
import scipy.misc
import sys
import threading
import time
import traceback

# FIXME(a.kazantsev): numpy.dot works 5 times faster with this option
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from veles.config import root, get_config
import veles.error as error
import veles.formats as formats
import veles.image as image
import veles.plotting_units as plotting_units
import veles.rnd as rnd
import veles.thread_pool as thread_pool
import veles.workflows as workflows
import veles.znicz.all2all as all2all
import veles.znicz.decision as decision
import veles.znicz.evaluator as evaluator
import veles.znicz.gd as gd
import veles.znicz.image_saver as image_saver
import veles.znicz.loader as loader

root.cache_fnme = get_config(
    root.cache_fnme, os.path.join(root.common.cache_dir, "channels.pickle"))

root.decision.fail_iterations = get_config(root.decision.fail_iterations, 1000)

root.decision.snapshot_prefix = get_config(root.decision.snapshot_prefix,
                                           "channles_108_24")

root.decision.use_dynamic_alpha = get_config(root.decision.use_dynamic_alpha,
                                             False)
root.export = get_config(root.export, False)
root.find_negative = get_config(root.find_negative, 0)
root.global_alpha = get_config(root.global_alpha, 0.01)
root.global_lambda = get_config(root.global_lambda, 0.00005)
root.grayscale = get_config(root.grayscale, False)
root.layers = get_config(root.layers, [108, 24])
root.loader.minibatch_size = get_config(root.loader.minibatch_size, 81)
root.loader.rect = get_config(root.loader.rect, (264, 129))
root.n_threads = get_config(root.n_threads, 32)

root.path_for_train_data = get_config(
    root.path_for_train_data, "/data/veles/channels/korean_960_540/train")

root.snapshot = get_config(root.snapshot, "")
root.validation_procent = get_config(root.validation_procent, 0.15)
root.weights_plotter.limit = get_config(root.weights_plotter.limit, 16)


class Loader(loader.FullBatchLoader):
    """Loads channels.
    """
    def __init__(self, workflow, **kwargs):
        channels_dir = kwargs.get("channels_dir", "")
        rect = kwargs.get("rect", (264, 129))
        grayscale = kwargs.get("grayscale", False)
        cache_fnme = kwargs.get("cache_fnme", "")
        kwargs["channels_dir"] = channels_dir
        kwargs["rect"] = rect
        kwargs["grayscale"] = grayscale
        kwargs["cache_fnme"] = cache_fnme
        super(Loader, self).__init__(workflow, **kwargs)
        # : Top-level configuration from channels_dir/conf.py
        self.top_conf_ = None
        # : Configuration from channels_dir/subdirectory/conf.py
        self.subdir_conf_ = {}
        self.channels_dir = root.path_for_train_data
        self.cache_fnme = root.cache_fnme
        self.rect = root.loader.rect
        self.grayscale = root.grayscale
        self.w_neg = None  # workflow for finding the negative dataset
        self.find_negative = root.find_negative
        self.channel_map = None
        self.pos = {}
        self.sz = {}
        self.file_map = {}  # sample index to its file name map
        self.attributes_for_cached_data = [
            "channels_dir", "rect", "channel_map", "pos", "sz",
            "class_samples", "grayscale", "file_map", "cache_fnme"]
        self.exports = ["rect", "pos", "sz"]

    def from_jp2(self, fnme):
        try:
            j2 = glymur.Jp2k(fnme)
        except:
            self.error("glymur.Jp2k() failed for %s" % (fnme))
            raise
        a2 = j2.read()
        if j2.box[2].box[1].colorspace == 16:  # RGB
            if self.grayscale:
                # Get Y component from RGB
                a = numpy.empty([a2.shape[0], a2.shape[1], 1],
                                dtype=numpy.uint8)
                a[:, :, 0:1] = numpy.clip(
                    0.299 * a2[:, :, 0:1] +
                    0.587 * a2[:, :, 1:2] +
                    0.114 * a2[:, :, 2:3], 0, 255)
                a = formats.reshape(a, [a2.shape[0], a2.shape[1]])
            else:
                # Convert to YUV
                # Y = 0.299 * R + 0.587 * G + 0.114 * B;
                # U = -0.14713 * R - 0.28886 * G + 0.436 * B + 128;
                # V = 0.615 * R - 0.51499 * G - 0.10001 * B + 128;
                # and transform to different planes
                a = numpy.empty([3, a2.shape[0], a2.shape[1]],
                                dtype=numpy.uint8)
                a[0:1, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = numpy.clip(
                    0.299 * a2[:, :, 0:1] +
                    0.587 * a2[:, :, 1:2] +
                    0.114 * a2[:, :, 2:3], 0, 255)
                a[1:2, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = numpy.clip(
                    (-0.14713) * a2[:, :, 0:1] +
                    (-0.28886) * a2[:, :, 1:2] +
                    0.436 * a2[:, :, 2:3] + 128, 0, 255)
                a[2:3, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = numpy.clip(
                    0.615 * a2[:, :, 0:1] +
                    (-0.51499) * a2[:, :, 1:2] +
                    (-0.10001) * a2[:, :, 2:3] + 128, 0, 255)
        elif j2.box[2].box[1].colorspace == 18:  # YUV
            if self.grayscale:
                a = numpy.empty([a2.shape[0], a2.shape[1], 1],
                                dtype=numpy.uint8)
                a[:, :, 0:1] = a2[:, :, 0:1]
                a = formats.reshape(a, [a2.shape[0], a2.shape[1]])
            else:
                # transform to different yuv planes
                a = numpy.empty([3, a2.shape[0], a2.shape[1]],
                                dtype=numpy.uint8)
                a[0:1, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 0:1]
                a[1:2, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 1:2]
                a[2:3, :, :].reshape(
                    a2.shape[0], a2.shape[1], 1)[:, :, 0:1] = a2[:, :, 2:3]
        else:
            raise error.ErrBadFormat("Unknown colorspace in %s" % (fnme))
        return a

    def sample_rect(self, a, pos, sz):
        if self.grayscale:
            aa = numpy.empty([self.rect[1], self.rect[0]], dtype=numpy.float32)
            x = a
            left = int(numpy.round(pos[0] * x.shape[1]))
            top = int(numpy.round(pos[1] * x.shape[0]))
            width = int(numpy.round(sz[0] * x.shape[1]))
            height = int(numpy.round(sz[1] * x.shape[0]))
            x = x[top:top + height, left:left + width].ravel().copy().\
                reshape((height, width), order="C")
            x = image.resize(x, self.rect[0], self.rect[1])
            aa[:] = x[:]
        else:
            aa = numpy.empty([3, self.rect[1], self.rect[0]],
                             dtype=numpy.float32)
            # Loop by color planes.
            for j in range(0, a.shape[0]):
                x = a[j]
                left = int(numpy.round(pos[0] * x.shape[1]))
                top = int(numpy.round(pos[1] * x.shape[0]))
                width = int(numpy.round(sz[0] * x.shape[1]))
                height = int(numpy.round(sz[1] * x.shape[0]))
                x = x[top:top + height, left:left + width].ravel().copy().\
                    reshape((height, width), order="C")
                x = image.resize(x, self.rect[0], self.rect[1])
                aa[j] = x

        if self.grayscale:
            formats.normalize(aa)
        else:
            # Normalize Y and UV planes separately.
            formats.normalize(aa[0])
            formats.normalize(aa[1:])

        return aa

    def append_sample(self, sample, lbl, fnme, n_negative, data_lock):
        data_lock.acquire()
        self.original_data.append(sample)
        self.original_labels.append(lbl)
        ii = len(self.original_data) - 1
        self.file_map[ii] = fnme
        if n_negative is not None:
            n_negative[0] += 1
        data_lock.release()
        return ii

    def from_jp2_async(self, fnme, pos, sz, data_lock, stat_lock,
                       i_sample, lbl, n_files, total_files,
                       n_negative, rand):
        """Loads, crops and normalizes image in the parallel thread.
        """
        a = self.from_jp2(fnme)

        sample = self.sample_rect(a, pos, sz)
        self.append_sample(sample, lbl, fnme, None, data_lock)

        # Collect negative dataset from positive samples only
        if lbl and self.w_neg is not None and self.find_negative > 0:
            # Sample pictures at random positions
            samples = numpy.zeros([self.find_negative, sample.size],
                                  dtype=self.w_neg[0][0].dtype)
            for i in range(self.find_negative):
                t = rand.randint(2)
                if t == 0:
                    # Sample vertical line
                    p = [pos[0] + (1 if pos[0] < 0.5 else -1) * sz[0],
                         rand.rand() * (1.0 - sz[1])]
                elif t == 1:
                    # Sample horizontal line
                    p = [rand.rand() * (1.0 - sz[0]),
                         pos[1] + (1 if pos[1] < 0.5 else -1) * sz[1]]
                else:
                    continue
                samples[i][:] = self.sample_rect(a, p, sz).ravel()[:]
            ll = self.get_labels_from_samples(samples)
            for i, l in enumerate(ll):
                if l == 0:
                    continue
                # negative found
                s = samples[i].reshape(sample.shape)
                ii = self.append_sample(s, 0, fnme, n_negative, data_lock)
                dirnme = "%s/found_negative_images" % (root.common.cache_dir)
                try:
                    os.mkdir(dirnme)
                except OSError:
                    pass
                fnme = "%s/0_as_%d.%d.png" % (dirnme, l, ii)
                scipy.misc.imsave(fnme, self.as_image(s))

        stat_lock.acquire()
        n_files[0] += 1
        if n_files[0] % 10 == 0:
            self.info("Read %d files (%.2f%%)" % (
                n_files[0], 100.0 * n_files[0] / total_files))
        stat_lock.release()

    def get_labels_from_samples(self, samples):
        weights = self.w_neg[0]
        bias = self.w_neg[1]
        n = len(weights)
        a = samples
        for i in range(n):
            a = numpy.dot(a, weights[i].transpose())
            a += bias[i]
            if i < n - 1:
                a *= 0.6666
                numpy.tanh(a, a)
                a *= 1.7159
        return a.argmax(axis=1)

    def get_label(self, dirnme):
        lbl = self.channel_map[dirnme].get("lbl")
        if lbl is None:
            lbl = int(dirnme)
        return lbl

    def load_data(self):
        if self.original_data is not None and self.original_labels is not None:
            return

        cached_data_fnme = (
            os.path.join(
                root.common.cache_dir,
                "%s_%s.pickle" %
                (os.path.basename(__file__), self.__class__.__name__))
            if not len(self.cache_fnme) else self.cache_fnme)
        self.info("Will try to load previously cached data from " +
                  cached_data_fnme)
        save_to_cache = True
        try:
            fin = open(cached_data_fnme, "rb")
            obj = pickle.load(fin)
            if obj["channels_dir"] != self.channels_dir:
                save_to_cache = False
                self.info("different dir found in cached data: %s" % (
                    obj["channels_dir"]))
                fin.close()
                raise FileNotFoundError()
            for k, v in obj.items():
                if type(v) == list:
                    o = self.__dict__[k]
                    if o is None:
                        o = []
                        self.__dict__[k] = o
                    del o[:]
                    o.extend(v)
                elif type(v) == dict:
                    o = self.__dict__[k]
                    if o is None:
                        o = {}
                        self.__dict__[k] = o
                    o.update(v)
                else:
                    self.__dict__[k] = v

            for k in self.pos.keys():
                self.info("%s: pos=(%.6f, %.6f) sz=(%.6f, %.6f)" % (
                    k, self.pos[k][0], self.pos[k][1],
                    self.sz[k][0], self.sz[k][1]))
            self.info("rect: (%d, %d)" % (self.rect[0], self.rect[1]))

            self.shuffled_indexes = pickle.load(fin)
            self.original_labels = pickle.load(fin)
            sh = ([self.rect[1], self.rect[0]] if self.grayscale
                  else [3, self.rect[1], self.rect[0]])
            n = int(numpy.prod(sh))
            # Get raw array from file
            self.original_data = []
            store_negative = self.w_neg is not None and self.find_negative > 0
            old_file_map = []
            n_not_exists_anymore = 0
            for i in range(len(self.original_labels)):
                a = numpy.fromfile(fin, dtype=numpy.float32, count=n)
                if store_negative:
                    if self.original_labels[i]:
                        del a
                        continue
                    if not os.path.isfile(self.file_map[i]):
                        n_not_exists_anymore += 1
                        del a
                        continue
                    old_file_map.append(self.file_map[i])
                self.original_data.append(a.reshape(sh))
            self.rnd[0].state = pickle.load(fin)
            fin.close()
            self.info("Succeeded")
            self.info("class_samples=[%s]" % (
                ", ".join(str(x) for x in self.class_samples)))
            if not store_negative:
                return
            self.info("Will search for a negative set at most %d "
                      "samples per image" % (self.find_negative))
            # Saving the old negative set
            self.info("Extracting the old negative set")
            self.file_map.clear()
            for i, fnme in enumerate(old_file_map):
                self.file_map[i] = fnme
            del old_file_map
            n = len(self.original_data)
            self.original_labels = list(0 for i in range(n))
            self.shuffled_indexes = None
            self.info("Done (%d extracted, %d not exists anymore)" % (
                n, n_not_exists_anymore))
        except FileNotFoundError:
            self.info("Failed")
            self.original_labels = []
            self.original_data = []
            self.shuffled_indexes = None
            self.file_map.clear()

        self.info("Will load data from original jp2 files")

        # Read top-level configuration
        try:
            fin = open(os.path.join(root.path_for_train_data, "conf.py"), "r")
            s = fin.read()
            fin.close()
            self.top_conf_ = {}
            exec(s, self.top_conf_, self.top_conf_)
        except:
            self.error("Error while executing %s/conf.py" % (
                self.channels_dir))
            raise

        # Read subdirectories configurations
        self.subdir_conf_.clear()
        for subdir in self.top_conf_["dirs_to_scan"]:
            try:
                fin = open("%s/%s/conf.py" % (self.channels_dir, subdir), "r")
                s = fin.read()
                fin.close()
                self.subdir_conf_[subdir] = {}
                exec(s, self.subdir_conf_[subdir], self.subdir_conf_[subdir])
            except:
                self.error("Error while executing %s/%s/conf.py" % (
                    self.channels_dir, subdir))
                raise

        # Parse configs
        self.channel_map = self.top_conf_["channel_map"]
        pos = {}
        rpos = {}
        sz = {}
        for subdir, subdir_conf in self.subdir_conf_.items():
            frame = subdir_conf["frame"]
            if subdir not in pos.keys():
                pos[subdir] = frame.copy()  # bottom-right corner
                rpos[subdir] = [0, 0]
            for pos_size in subdir_conf["channel_map"].values():
                pos[subdir][0] = min(pos[subdir][0], pos_size["pos"][0])
                pos[subdir][1] = min(pos[subdir][1], pos_size["pos"][1])
                rpos[subdir][0] = max(rpos[subdir][0],
                                      pos_size["pos"][0] + pos_size["size"][0])
                rpos[subdir][1] = max(rpos[subdir][1],
                                      pos_size["pos"][1] + pos_size["size"][1])
            # Convert to relative values
            pos[subdir][0] /= frame[0]
            pos[subdir][1] /= frame[1]
            rpos[subdir][0] /= frame[0]
            rpos[subdir][1] /= frame[1]
            sz[subdir] = [rpos[subdir][0] - pos[subdir][0],
                          rpos[subdir][1] - pos[subdir][1]]

        self.info("Found rectangles:")
        for k in pos.keys():
            self.info("%s: pos=(%.6f, %.6f) sz=(%.6f, %.6f)" % (
                k, pos[k][0], pos[k][1], sz[k][0], sz[k][1]))

        self.info("Adjusted rectangles:")
        for k in pos.keys():
            # sz[k][0] *= 1.01
            # sz[k][1] *= 1.01
            pos[k][0] += (rpos[k][0] - pos[k][0] - sz[k][0]) * 0.5
            pos[k][1] += (rpos[k][1] - pos[k][1] - sz[k][1]) * 0.5
            pos[k][0] = min(pos[k][0], 1.0 - sz[k][0])
            pos[k][1] = min(pos[k][1], 1.0 - sz[k][1])
            pos[k][0] = max(pos[k][0], 0.0)
            pos[k][1] = max(pos[k][1], 0.0)
            self.info("%s: pos=(%.6f, %.6f) sz=(%.6f, %.6f)" % (
                k, pos[k][0], pos[k][1], sz[k][0], sz[k][1]))

        self.pos.clear()
        self.pos.update(pos)
        self.sz.clear()
        self.sz.update(sz)

        max_lbl = 0
        files = {}
        total_files = 0
        baddir = re.compile("bad", re.IGNORECASE)
        jp2 = re.compile("\.jp2$", re.IGNORECASE)
        for subdir, subdir_conf in self.subdir_conf_.items():
            for dirnme in subdir_conf["channel_map"].keys():
                max_lbl = max(max_lbl, self.get_label(dirnme))
                relpath = "%s/%s" % (subdir, dirnme)
                found_files = []
                fordel = []
                for basedir, dirlist, filelist in os.walk(
                        "%s/%s" % (self.channels_dir, relpath)):
                    for i, nme in enumerate(dirlist):
                        if baddir.search(nme) is not None:
                            fordel.append(i)
                    while len(fordel) > 0:
                        dirlist.pop(fordel.pop())
                    for nme in filelist:
                        if jp2.search(nme) is not None:
                            found_files.append("%s/%s" % (basedir, nme))
                found_files.sort()
                files[relpath] = found_files
                total_files += len(found_files)
        self.info("Found %d files" % (total_files))

        # Read samples in parallel
        rand = rnd.Rand()
        rand.seed(numpy.fromfile("/dev/urandom", dtype=numpy.int32,
                                 count=1024))
        # FIXME(a.kazantsev): numpy.dot is thread-safe with this value
        # on ubuntu 13.10 (due to the static number of buffers in libopenblas)
        n_threads = root.n_threads
        pool = thread_pool.ThreadPool(minthreads=1, maxthreads=n_threads,
                                      queue_size=n_threads)
        data_lock = threading.Lock()
        stat_lock = threading.Lock()
        n_files = [0]
        n_negative = [0]
        i_sample = 0
        for subdir in sorted(self.subdir_conf_.keys()):
            subdir_conf = self.subdir_conf_[subdir]
            for dirnme in sorted(subdir_conf["channel_map"].keys()):
                relpath = "%s/%s" % (subdir, dirnme)
                self.info("Will load from %s" % (relpath))
                lbl = self.get_label(dirnme)
                for fnme in files[relpath]:
                    pool.request(self.from_jp2_async, (
                        fnme, pos[subdir], sz[subdir],
                        data_lock, stat_lock,
                        0 + i_sample, 0 + lbl, n_files, total_files,
                        n_negative, rand))
                    i_sample += 1
        pool.shutdown(execute_remaining=True)

        if (len(self.original_data) != len(self.original_labels) or
                len(self.file_map) != len(self.original_labels)):
            raise Exception("Logic error")

        if self.w_neg is not None and self.find_negative > 0:
            n_positive = numpy.count_nonzero(self.original_labels)
            self.info("Found %d negative samples (%.2f%%)" % (
                n_negative[0], 100.0 * n_negative[0] / n_positive))

        self.info("Loaded %d samples with resize and %d without" % (
            image.resize_count, image.asitis_count))

        self.class_samples[0] = 0
        self.class_samples[1] = 0
        self.class_samples[2] = len(self.original_data)

        # Randomly generate validation set from train.
        self.info("Will extract validation set from train")
        self.extract_validation_from_train(rand=rnd.default2)

        # Saving all the samples
        """
        self.info("Dumping all the samples to %s" % (root.common.cache_dir))
        for i in self.shuffled_indexes:
            l = self.original_labels[i]
            dirnme = "%s/%03d" % (root.common.cache_dir, l)
            try:
                os.mkdir(dirnme)
            except OSError:
                pass
            fnme = "%s/%d.png" % (dirnme, i)
            scipy.misc.imsave(fnme, self.as_image(self.original_data[i]))
        self.info("Done")
        """

        self.info("class_samples=[%s]" % (
            ", ".join(str(x) for x in self.class_samples)))

        if not save_to_cache:
            return
        self.info("Saving loaded data for later faster load to "
                  "%s" % (cached_data_fnme))
        fout = open(cached_data_fnme, "wb")
        obj = {}
        for name in self.attributes_for_cached_data:
            obj[name] = self.__dict__[name]
        pickle.dump(obj, fout)
        pickle.dump(self.shuffled_indexes, fout)
        pickle.dump(self.original_labels, fout)
        # Because pickle doesn't support greater than 4Gb arrays
        for i in range(len(self.original_data)):
            self.original_data[i].ravel().tofile(fout)
        # Save random state
        pickle.dump(self.rnd[0].state, fout)
        fout.close()
        self.info("Done")

    def as_image(self, x):
        if len(x.shape) == 2:
            x = x.copy()
        elif len(x.shape) == 3:
            if x.shape[2] == 3:
                x = x.copy()
            elif x.shape[0] == 3:
                xx = numpy.empty([x.shape[1], x.shape[2], 3],
                                 dtype=x.dtype)
                xx[:, :, 0:1] = x[0:1, :, :].reshape(
                    x.shape[1], x.shape[2], 1)[:, :, 0:1]
                xx[:, :, 1:2] = x[1:2, :, :].reshape(
                    x.shape[1], x.shape[2], 1)[:, :, 0:1]
                xx[:, :, 2:3] = x[2:3, :, :].reshape(
                    x.shape[1], x.shape[2], 1)[:, :, 0:1]
                x = xx
            else:
                raise error.ErrBadFormat()
        else:
            raise error.ErrBadFormat()
        return formats.norm_image(x, True)


class Workflow(workflows.OpenCLWorkflow):
    """Workflow.
    """
    def __init__(self, workflow, **kwargs):
        layers = kwargs.get("layers")
        device = kwargs.get("device")
        kwargs["layers"] = layers
        kwargs["device"] = device
        super(Workflow, self).__init__(workflow, **kwargs)

        self.saver = None

        self.rpt.link_from(self.start_point)

        self.loader = Loader(self)
        self.loader.link_from(self.rpt)

        # Add forward units
        del self.forward[:]
        for i in range(0, len(layers)):
            if i < len(layers) - 1:
                aa = all2all.All2AllTanh(self, output_shape=[layers[i]],
                                         device=device)
            else:
                aa = all2all.All2AllSoftmax(self, output_shape=[layers[i]],
                                            device=device)
            self.forward.append(aa)
            if i:
                self.forward[i].link_from(self.forward[i - 1])
                self.forward[i].input = self.forward[i - 1].output
            else:
                self.forward[i].link_from(self.loader)
                self.forward[i].input = self.loader.minibatch_data

        # Add Image Saver unit
        self.image_saver = image_saver.ImageSaver(self, yuv=True)
        self.image_saver.link_from(self.forward[-1])
        self.image_saver.input = self.loader.minibatch_data
        self.image_saver.output = self.forward[-1].output
        self.image_saver.max_idx = self.forward[-1].max_idx
        self.image_saver.indexes = self.loader.minibatch_indexes
        self.image_saver.labels = self.loader.minibatch_labels
        self.image_saver.minibatch_class = self.loader.minibatch_class
        self.image_saver.minibatch_size = self.loader.minibatch_size

        # Add evaluator for single minibatch
        self.ev = evaluator.EvaluatorSoftmax(self, device=device,
                                             compute_confusion_matrix=False)
        self.ev.link_from(self.image_saver)
        self.ev.y = self.forward[-1].output
        self.ev.batch_size = self.loader.minibatch_size
        self.ev.labels = self.loader.minibatch_labels
        self.ev.max_idx = self.forward[-1].max_idx
        self.ev.max_samples_per_epoch = self.loader.total_samples

        # Add decision unit
        self.decision = decision.Decision(
            self, snapshot_prefix=root.decision.snapshot_prefix,
            use_dynamic_alpha=root.decision.use_dynamic_alpha,
            fail_iterations=root.decision.fail_iterations)
        self.decision.link_from(self.ev)
        self.decision.minibatch_class = self.loader.minibatch_class
        self.decision.minibatch_last = self.loader.minibatch_last
        self.decision.minibatch_n_err = self.ev.n_err
        # self.decision.minibatch_confusion_matrix = self.ev.confusion_matrix
        self.decision.class_samples = self.loader.class_samples

        self.image_saver.gate_skip = ~self.decision.just_snapshotted
        self.image_saver.this_save_time = self.decision.snapshot_time

        # Add gradient descent units
        del self.gd[:]
        self.gd.extend(None for i in range(0, len(self.forward)))
        self.gd[-1] = gd.GDSM(self, device=device)
        # self.gd[-1].link_from(self.decision)
        self.gd[-1].err_y = self.ev.err_y
        self.gd[-1].y = self.forward[-1].output
        self.gd[-1].h = self.forward[-1].input
        self.gd[-1].weights = self.forward[-1].weights
        self.gd[-1].bias = self.forward[-1].bias
        self.gd[-1].gate_skip = self.decision.gd_skip
        self.gd[-1].batch_size = self.loader.minibatch_size
        for i in range(len(self.forward) - 2, -1, -1):
            self.gd[i] = gd.GDTanh(self, device=device)
            self.gd[i].link_from(self.gd[i + 1])
            self.gd[i].err_y = self.gd[i + 1].err_h
            self.gd[i].y = self.forward[i].output
            self.gd[i].h = self.forward[i].input
            self.gd[i].weights = self.forward[i].weights
            self.gd[i].bias = self.forward[i].bias
            self.gd[i].gate_skip = self.decision.gd_skip
            self.gd[i].batch_size = self.loader.minibatch_size
        self.rpt.link_from(self.gd[0])

        self.end_point.link_from(self.decision)
        self.end_point.gate_block = ~self.decision.complete

        self.loader.gate_block = self.decision.complete

        # Error plotter
        self.plt = []
        styles = ["r-", "b-", "k-"]
        for i in range(1, 3):
            self.plt.append(plotting_units.AccumulatingPlotter(
                self, name="num errors", plot_style=styles[i],
                ylim=(0, 100)))
            self.plt[-1].input = self.decision.epoch_n_err_pt
            self.plt[-1].input_field = i
            self.plt[-1].link_from(self.decision)
            self.plt[-1].gate_block = ~self.decision.epoch_ended
        self.plt[0].clear_plot = True
        self.plt[-1].redraw_plot = True
        # Weights plotter
        self.decision.vectors_to_sync[self.gd[0].weights] = 1
        self.plt_w = plotting_units.Weights2D(
            self, name="First Layer Weights", limit=root.weights_plotter.limit,
            yuv=True)
        self.plt_w.input = [self.gd[0].weights.v]
        self.plt_w.get_shape_from = self.forward[0].input
        self.plt_w.input_field = 0
        self.plt_w.link_from(self.decision)
        self.plt_w.gate_block = ~self.decision.epoch_ended
        # Image plottter
        self.decision.vectors_to_sync[self.forward[0].input] = 1
        self.decision.vectors_to_sync[self.ev.labels] = 1
        self.plt_i = plotting_units.Image(self, name="Input", yuv=True)
        self.plt_i.inputs.append(self.decision.sample_label)
        self.plt_i.input_fields.append(0)
        self.plt_i.inputs.append(self.decision.sample_input)
        self.plt_i.input_fields.append(0)
        self.plt_i.link_from(self.decision)
        self.plt_i.gate_block = ~self.decision.epoch_ended
        # Confusion matrix plotter
        """
        self.plt_mx = []
        for i in range(1, 3):
            self.plt_mx.append(plotters.MatrixPlotter(
                self, name=(("Test", "Validation", "Train")[i] + " matrix")))
            self.plt_mx[-1].input = self.decision.confusion_matrixes
            self.plt_mx[-1].input_field = i
            self.plt_mx[-1].link_from(self.decision)
            self.plt_mx[-1].gate_block = ~self.decision.epoch_ended
        self.gd[-1].link_from(self.plt_mx[-1])
        """
        self.gd[-1].link_from(self.decision)

    def initialize(self, global_alpha, global_lambda, minibatch_maxsize,
                   w_neg, device):
        self.loader.minibatch_maxsize[0] = minibatch_maxsize
        self.loader.w_neg = w_neg
        self.ev.device = device
        for g in self.gd:
            g.device = device
            g.global_alpha = global_alpha
            g.global_lambda = global_lambda
        for forward in self.forward:
            forward.device = device
        return super(Workflow, self).initialize()


def run(load, main):
    layers = []
    for s in root.layers:
        layers.append(int(s))
    logging.info("Will train NN with layers: %s"
                 % (" ".join(str(x) for x in layers)))

    w_neg = None
    try:
        w, _ = load(Workflow, layers=root.layers)
        if root.export:
            tm = time.localtime()
            s = "%d.%02d.%02d_%02d.%02d.%02d" % (
                tm.tm_year, tm.tm_mon, tm.tm_mday,
                tm.tm_hour, tm.tm_min, tm.tm_sec)
            fnme = os.path.join(root.common.snapshot_dir,
                                "channels_workflow_%s" % s)
            try:
                w.export(fnme)
                logging.info("Exported successfully to %s.tar.gz" % (fnme))
            except:
                a, b, c = sys.exc_info()
                traceback.print_exception(a, b, c)
                logging.error("Error while exporting.")
            return
        if root.find_negative > 0:
            if type(w) != tuple or len(w) != 2:
                logging.error(
                    "Snapshot with weights and biases only "
                    "should be provided when find_negative is supplied. "
                    "Will now exit.")
                return
            w_neg = w
            raise IOError()
    except IOError:
        if root.export:
            logging.error("Valid snapshot should be provided if "
                          "export is True. Will now exit.")
            return
        if (root.find_negative > 0 and w_neg is None):
            logging.error("Valid snapshot should be provided if "
                          "find_negative supplied. Will now exit.")
            return
    fnme = (os.path.join(root.common.cache_dir, root.decision.snapshot_prefix)
            + ".txt")
    logging.info("Dumping file map to %s" % (fnme))
    fout = open(fnme, "w")
    file_map = w.loader.file_map
    for i in sorted(file_map.keys()):
        fout.write("%d\t%s\n" % (i, file_map[i]))
    fout.close()
    logging.info("Done")
    logging.info("Will execute workflow now")
    main(global_alpha=root.global_alpha, global_lambda=root.global_lambda,
         minibatch_maxsize=root.loader.minibatch_size, w_neg=w_neg)
