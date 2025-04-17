import os
import sys
import json
import cv2 as cv
import numpy as np
from dataset import Dataset
from pathlib import Path
from pycocotools import mask as pycmask

GRID_STEP = 30

assert len(sys.argv) == 3, "Usage: python train.py <path_to_labels.json> <path_to_imgs_folder>"

labels_path = Path(sys.argv[1]).resolve()
imgs_path = Path(sys.argv[2]).resolve()

assert os.path.exists(labels_path), f"Labels path does not exist: {labels_path}"
assert os.path.exists(imgs_path), f"Images path does not exist: {imgs_path}"

input_dataset = Dataset(imgs_path, labels_path)

# Form prompt point grid

first = input_dataset.get_all()[0]

_, height, width, _ = input_dataset.get_img_data(first)

y_coords = np.arange(0, height, GRID_STEP)
x_coords = np.arange(0, width, GRID_STEP)

grid = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

# Read in dataset and assign points to masks
data = []
for entry in input_dataset.get_all():
    
    img_name = entry['image']['image_id']
    height = entry['image']['height']
    width = entry['image']['width']

    annotations = entry['annotations']

    masks = []
    for ann in annotations:
        seg = ann['segmentation']
        counts = seg['counts']

        print(counts)

        counts = counts.decode('utf-8')
        seg['counts'] = counts

        mask = pycmask.decode(ann['segmentation'])

        cv.imshow('Mask', mask * 255)
        cv.waitKey(0)

cv.destroyAllWindows()