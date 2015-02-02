"""
Created on Jun 7, 2014

Loader package.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""

from veles.loader.base import ILoader, Loader, CLASS_NAME, TARGET, TRAIN, \
    VALID, TEST
from veles.loader.fullbatch import IFullBatchLoader, FullBatchLoader, \
    FullBatchLoaderMSE
from veles.loader.image import IImageLoader, ImageLoader, IFileImageLoader, \
    FileImageLoader
from veles.loader.fullbatch_image import FullBatchFileImageLoader
from veles.loader.pickles import PicklesImageFullBatchLoader
from veles.loader.saver import MinibatchesSaver, MinibatchesLoader
