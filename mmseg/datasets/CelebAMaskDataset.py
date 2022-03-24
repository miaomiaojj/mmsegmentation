import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class CelebAMaskDataset(CustomDataset):
    CLASSES = ('background', 'skin', 'nose', 'eye glasses', 'left eye', 'right eye', 'left brow', 'right brow','left ear','right ear ','mouth','upper lip ','lower lip','hair','hat','earring','necklace','neck','cloth')
    PALETTE = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    def __init__(self, **kwargs):
        super(CelebAMaskDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)