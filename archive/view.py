# view.py
# Gavin Haynes
# Confirm that the dataset is in the correct format and view annotations side-by-side.

import cv2 as cv
import glob
import numpy as np
from pathlib import Path

dataset_path = Path.cwd().parent.absolute()
imgs_folder = dataset_path.joinpath('imgs')
masks_folder = dataset_path.joinpath('masks')

print('Loading images from', imgs_folder)
print('Loading masks from', masks_folder)