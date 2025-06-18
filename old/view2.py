# view.py
# Gavin Haynes
# View dataset images along with their COCO annotations.

import os
from sys import argv
import json
import cv2 as cv
import numpy as np
from pathlib import Path
from pycocotools import mask as pycmask

def main():
    assert len(argv) == 2, "Usage: python view.py <path_to_dataset_folder>"

    data_path = Path(argv[1]).resolve() / 'labels.json'
    imgs_folder_path = data_path / 'imgs'

    data = []

    with open(data_path) as data_file:
        data = json.load(data_file)

    assert len(data) > 0, 'ERROR: No data found in ' + str(data_path)

    for entry in data:
        image_path, height, width, masks = get(entry)
        
        cv.imshow(image_path, draw_contours(image_path, masks))
        cv.waitKey(0)

def get(entry: dict):
    """
    Get data required to view an entry in the dataset.
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

def draw_contours(image_path: str, masks: list):
    """
    Display the image with the given masks.
    Args:
        image_path (str): Path to the image.
        masks (list): List of masks to display.
    """
    img = cv.imread(image_path)
    
    for mask in masks:
        mask *= 255
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        for contour in contours:
            cv.drawContours(img, contour, -1, (0, 0, 255), 2)

    return img

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()