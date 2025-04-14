# dataset.py
# Gavin Haynes

# I need things like 
# Read from label studio file
# Convert to coco
# Make coco entries when augmenting
# Fetching info to view

import os
import sys
import json
import cv2 as cv
import numpy as np
from labelconverter import rle_to_mask
from pathlib import Path
from pycocotools import mask as pycmask

def ls_to_coco_entry(img_path: str, results: list):
    """
    Conver a LS entry to COCO format.
    Args:
        img_path (str): Path to the image.
        results (list): List of segmentation results in LS format.
    Returns:
        dict: COCO dataset entry.
    """
    
    height = results[0]['original_height']
    width = results[0]['original_width']
    
    image_info = {
        'image_id': img_path,
        'width': width,
        'height': height,
        'file_name': img_path
    }

    # Create annotation entries for each mask in the results

    annotations = []
    
    for id, annotation in enumerate(results):

        coco_rle = convert_rle(annotation)
        json_rle = coco_rle.copy()
        json_rle['counts'] = json_rle['counts'].decode('utf-8')

        annotations.append({
            'id': id,
            'segmentation': json_rle,
            'bbox': pycmask.toBbox(coco_rle).tolist(),
            'area': int(pycmask.area(coco_rle)),
            'predicted_iou': 0.0,
            'stability_score': 0.0,
            'crop_box': pycmask.toBbox(coco_rle).tolist(),
            'point_coords': []
        })

    return {'image': image_info, 'annotations': annotations}

def convert_rle(label: list) -> dict:
    """
    Convert an RLE mask to COCO format.
    Args:
        label (list): Label-Studio RLE string).
    Returns:
        dict: COCO format RLE mask. 
    """
    
    mask = rle_to_mask(
        label['value']['rle'],
        label['original_height'],
        label['original_width']
    )

    mask[mask > 0] = 1
    mask = np.asfortranarray(mask, dtype=np.uint8)
       
    return pycmask.encode(mask)

def get_img_data(entry: dict):
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