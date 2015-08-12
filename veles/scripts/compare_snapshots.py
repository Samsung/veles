#!/usr/bin/env python3
# -*-coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on May 23, 2013

This script compares snapshots taken by :class:`veles.snapshotter.Snapshotter`
in the breadth-first tree traversal order. It prints relative differences
between contained :class:`veles.memory.Array` instances.

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


import argparse
import logging
import os

from veles.dot_pip import install_dot_pip
install_dot_pip()
import numpy
from veles.compat import from_none
from veles.external.prettytable import PrettyTable
from veles.logger import Logger
from veles.memory import Array
from veles.snapshotter import SnapshotterToFile


SORT_CHOICES = ("dep", "unit", "attr", "avgreldiff", "avgdiff", "maxdiff")
SORT_CHOICES_MAP = {k: i for i, k in enumerate(SORT_CHOICES)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Do not print logs.")
    parser.add_argument(
        "-s", "--sort", choices=SORT_CHOICES,
        help="Sort by this field (may be specified multiple times).",
        nargs='*', action='append', default=["dep", "avgreldiff"])
    parser.add_argument('first', help='Path to the first snapshot.')
    parser.add_argument('second', help="Path to the second snapshot.")
    return parser.parse_args()


def load_snapshot(path):
    try:
        return SnapshotterToFile.import_(path)
    except Exception as e:
        logging.critical("Failed to load the snapshot at %s", path)
        raise from_none(e)


def get_diffs(first_units, second_units):
    for index, (first_unit, second_unit) in enumerate(zip(first_units,
                                                          second_units)):
        for key, first_val in first_unit.__dict__.items():
            if not isinstance(first_val, Array):
                continue
            second_val = getattr(second_unit, key)
            assert isinstance(second_val, Array)
            if first_val.mem is None:
                assert second_val.mem is None
                continue
            diff = first_val.mem - second_val.mem
            avg_diff = numpy.mean(numpy.abs(diff), dtype=numpy.float64)
            val_sum = first_val.mem + second_val.mem
            nz = numpy.nonzero(val_sum)
            rel = 2 * (diff[nz] / val_sum[nz])
            if len(rel) > 0:
                avg_rel_diff = numpy.mean(numpy.abs(rel), dtype=numpy.float64)
            else:
                avg_rel_diff = float(not (diff == 0).all())
            max_diff = numpy.max(numpy.abs(diff))
            yield index, first_unit.name, key, avg_rel_diff, avg_diff, max_diff


def sort_diffs(diffs, sorting):
    diffs = list(diffs)

    def sort_key(record):
        return tuple(record[SORT_CHOICES_MAP[sk]] for sk in sorting)

    diffs.sort(key=sort_key)
    return diffs


def print_table(diffs):
    table = PrettyTable("Unit", "Attribute", "Average Relative Diff",
                        "Average Diff", "Max Diff")
    for fn in table.field_names[:2]:
        table.align[fn] = "l"
    for fn in table.field_names[2:]:
        table.align[fn] = "c"
    for diff in diffs:
        table.add_row(*diff[1:])
    print(table)


def main():
    args = parse_args()
    Logger.setup_logging(logging.INFO if not args.quiet else logging.WARNING)
    logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
    logger.info("Loading snapshots...")
    first, second = tuple(load_snapshot(p) for p in (args.first, args.second))
    if (type(first) != type(second) or  # pylint: disable=W0622
            first.checksum != second.checksum):
        raise ValueError("Cannot compare different workflows")
    logger.info("Comparing snapshots...")
    diffs = list(get_diffs(first.units_in_dependency_order,
                           second.units_in_dependency_order))
    logger.info("Sorting the results...")
    diffs = sort_diffs(diffs, args.sort)
    logger.info("Printing the results...")
    print_table(diffs)

if __name__ == "__main__":
    main()
