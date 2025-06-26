# ratdataset.py
# Interface to the RAT dataset, which contains image / mask pairs.

import cv2 as cv
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class RatDataset(Dataset):
    def __init__(self, imgs_dir: Path, masks_dir: Path):
        self.img_dir = imgs_dir
        self.masks_dir = masks_dir
        self.img_paths = list(imgs_dir.glob('*'))

    def __len__(self):
        return len(list(self.img_paths))

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx):
        """
        Fetch an image and its corresponding masks.
        """
        img_path = self.img_paths[idx]
        mask_path: Path = self.masks_dir / (img_path.stem + '.npy')
        img: np.ndarray = cv.imread(str(img_path))
        mask: np.ndarray = np.load(mask_path)
        
        return img, mask