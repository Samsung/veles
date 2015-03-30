"""
Created on Jun 7, 2014

Loader package.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
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
