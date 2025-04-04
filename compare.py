# compare.py
# Gavin Haynes
# View input images along with annotations.

import os
import cv2 as cv
import numpy
import json
from itertools import groupby
from pycocotools import mask as mask_utils

mask = cv.imread('./dataset/masks/image-1-mask-0.png', cv.COLOR_RGB2GRAY)

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

print(binary_mask_to_rle(mask))