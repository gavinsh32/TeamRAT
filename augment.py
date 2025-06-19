# augment.py
# Apply pre-determined augmentations to training data, also scaling the volume by a factor.

import albumentations as A
import cv2 as cv
import numpy as np
from pathlib import Path