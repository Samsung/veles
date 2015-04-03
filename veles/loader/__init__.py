# -*- coding: utf-8 -*-
"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

Created on Jun 7, 2014

Loader package

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


from veles.loader.base import ILoader, Loader, CLASS_NAME, TARGET, TRAIN, \
    VALID, TEST
from veles.loader.fullbatch import IFullBatchLoader, FullBatchLoader, \
    FullBatchLoaderMSE
from veles.loader.file_loader import IFileLoader
try:
    from veles.loader.image import IImageLoader, ImageLoader
    from veles.loader.file_image import FileImageLoader, \
        AutoLabelFileImageLoader
    from veles.loader.fullbatch_image import FullBatchFileImageLoader, \
        FullBatchFileImageLoaderMSE, FullBatchAutoLabelFileImageLoader
    from veles.loader.pickles import PicklesImageFullBatchLoader
except ImportError as e:
    import warnings
    warnings.warn("Image loaders will be unavailable: %s" % e)
    IImageLoader = ImageLoader = AutoLabelFileImageLoader = FileImageLoader = \
        FullBatchFileImageLoader = FullBatchFileImageLoaderMSE = \
        FullBatchAutoLabelFileImageLoader = PicklesImageFullBatchLoader = \
        object

from veles.loader.saver import MinibatchesSaver, MinibatchesLoader
