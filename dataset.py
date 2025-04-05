# dataset.py
# Gavin Haynes
# Read in Label Studio Annotations, fetch labels, and check your dataset.

import json
import labelconverter as lc
import numpy as np
import cv2 as cv
import os
from pathlib import Path


class Dataset:

    def __init__(self, dataset_folder: Path):

        dataset_folder = dataset_folder.resolve()
        imgs_folder = dataset_folder / 'imgs'

        # Read in all image paths
        self.img_paths = [path for path in imgs_folder.glob('*')]

        # Check that there are images in the image folder
        if len(self.img_paths) < 1:
            print('ERROR: found no images in', imgs_folder)

        # Seek values for iterating through the JSON file
        self.taskNum = 0
        self.labelNum = 0

        # Read in essential data from labels
        labels_file = dataset_folder / 'labels.json'
        with open(labels_file) as data_file:
            self.data = json.load(data_file)

    def check(self, imgs_path: Path):
        pass
    
    # Load JSON File and store the dict
    def loadJSON(self, json_path: Path) -> bool:
        return True

dataset_folder = Path('./dataset').resolve()

print(dataset_folder)

dataset = Dataset(dataset_folder)