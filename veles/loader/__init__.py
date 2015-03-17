"""
Created on Jun 7, 2014

Loader package.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""
from veles.compat import from_none, PYPY

from veles.loader.base import ILoader, Loader, CLASS_NAME, TARGET, TRAIN, \
    VALID, TEST
from veles.loader.fullbatch import IFullBatchLoader, FullBatchLoader, \
    FullBatchLoaderMSE
try:
    from veles.loader.image import IImageLoader, ImageLoader
    from veles.loader.file_image import IFileImageLoader, FileImageLoader
    from veles.loader.fullbatch_image import FullBatchFileImageLoader, \
        FullBatchFileImageLoaderMSE, FullBatchAutoLabelFileImageLoader
    from veles.loader.pickles import PicklesImageFullBatchLoader
except ImportError as e:
    if not PYPY:
        raise from_none(e)
    IImageLoader = ImageLoader = IFileImageLoader = FileImageLoader = \
        FullBatchFileImageLoader = FullBatchFileImageLoaderMSE = \
        FullBatchAutoLabelFileImageLoader = PicklesImageFullBatchLoader = \
        object

from veles.loader.saver import MinibatchesSaver, MinibatchesLoader
