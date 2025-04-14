# dataset.py
# Gavin Haynes
# Working with COCO dataset

import os
import sys
import json
import cv2 as cv
import numpy as np
from labelconverter import rle_to_mask
from pathlib import Path
from pycocotools import mask as pycmask

"""
Dataset Format:
{
    "image"                 : image_info,
    "annotations"           : [annotation],
}

image_info {
    "image_id"              : int,              # Image id
    "width"                 : int,              # Image width
    "height"                : int,              # Image height
    "file_name"             : str,              # Image filename
}

annotation {
    "id"                    : int,              # Annotation id
    "segmentation"          : dict,             # Mask saved in COCO RLE format.
    "bbox"                  : [x, y, w, h],     # The box around the mask, in XYWH format
    "area"                  : int,              # The area in pixels of the mask
    "predicted_iou"         : float,            # The model's own prediction of the mask's quality
    "stability_score"       : float,            # A measure of the mask's quality
    "crop_box"              : [x, y, w, h],     # The crop of the image used to generate the mask, in XYWH format
    "point_coords"          : [[x, y]],         # The point coordinates input to the model to generate the mask
}
"""

class Dataset:
    def __init__(self, imgs_folder: Path, data_path: Path):
        """
        Initialize the Dataset class.
        Args:
            imgs_folder (str): Path to the folder containing images.
            data_path (str): Path to the JSON file containing annotations.
        """

        # Check for images folder.
        self.imgs_folder = imgs_folder.resolve()
        assert os.path.exists(self.imgs_folder), \
            f"Images folder does not exist: {self.imgs_folder}"
        
        # Check for the labels file.
        self.labels = data_path.resolve()
        assert os.path.exists(self.labels), \
            f"Data path does not exist: {self.labels}"
        
        # Load in labels json and check that there is data.
        with open(self.labels) as data_file:
            self.data = json.load(data_file)
            assert len(self.data) > 0, 'ERROR: No data found in ' + str(self.data_path)

    def get_all(self):
        return self.data
    
    def get_img_data(self, entry: dict):
        """
        Get data required to view from an entry in the dataset.
        Args:
            entry (dict): Entry in the dataset.
        Returns:
            tuple: Image path, image width, image height, and segmentation masks.
        """

        masks = []

        for ann in entry['annotations']:
            seg = ann['segmentation']
            seg['counts'] = bytes(seg['counts'], 'utf-8')
            masks.append(pycmask.decode(seg))

        return (
            entry['image']['file_name'],
            entry['image']['height'],
            entry['image']['width'],
            masks
        )

    def __len__(self):
        return len(self.data)

    def draw_contours(self, img, masks: list):
        """
        Display the image with the given masks.
        Args:
            image_path (str): Path to the image.
            masks (list): List of masks to display.
        """
    
        for mask in masks:
            mask *= 255
            contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            for contour in contours:
                cv.drawContours(img, contour, -1, (0, 0, 255), 2)

        return img