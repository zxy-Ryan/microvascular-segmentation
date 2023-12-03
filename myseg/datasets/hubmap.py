# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from myseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class HUBMAPDataset(BaseSegDataset):
    """HRF dataset.

    In segmentation map annotation for HRF, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('background', 'blood_vessel', 'glomerulus'),
        palette=[[255, 255, 255], [120, 120, 120], [6, 230, 230]])

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)
