# dataset.py
# Gavin Haynes
# Read in Label Studio Annotations, fetch labels, and check your dataset.

import json                     # Reading annotations in
import numpy as np              # Mask and image data
import os                       # System operations
from pathlib import Path        # Path manipulation
import pycocotools.mask as mask # Mask re-encoding
import cv2 as cv
import labelconverter as lc     # See labelconvert.py, I did not write this
import albumentations as A
from enum import Enum           

# Internal Dataset Format: dict[image_name: dict[width,height,rle]]
# Stores only essential data regarding the dataset required for manipulating
# RLE masks.

class Dataset:
    """
    Helps with Label Studio Segmentation Masks. Convert, augment, and then export to COCO format.
    """

    # Read in dataset and load valid images and annotations
    def __init__(self, dataset_folder_path: str):
        """
        Reads in a dataset and stores a cleaned internal copy.
        Args:
            dataset_folder_path: name of dataset folder
        Returns: None
        """

        dataset_folder = Path(dataset_folder_path).resolve()
        imgs_folder = dataset_folder / 'imgs'
        labels_file = dataset_folder / 'labels.json'
        
        self.data = {}      # Keys: image path, Content: dict with img data
        data = None         # Original JSON file     
        
        # Open JSON file and read in
        with open(labels_file) as data_file:
            data = json.load(data_file)

        # Loop through each task, one set of annotations per image
        for task in data:   

            # Grab name of the source image that was annotated, and trim
            img_name = task['file_upload'].split('-')[1]
            img_path = str(imgs_folder / img_name)

            # Only read in annotations if there is a matching image
            if os.path.exists(img_path):
                
                # List of masks corresponding to an image
                results = task['annotations'][0]['result']
                width = results[0]['original_width']
                height = results[0]['original_height']
                coco_rles = []

                # Convert Label Studio RLE to COCO RLE
                for result in results:
                    rle = result['value']['rle']
                    arr = lc.rle_to_mask(rle, width, height)
                    arr[arr > 0] = 1
                    arr = np.asfortranarray(arr, dtype=np.uint8)
                    coco_rles.append(mask.encode(arr))

                # Create a new entry with essential information   
                self.insert(img_path, width, height, coco_rles)

    # Insert an image with annotations to internal dataset
    def insert(self, key: str, width: int, height: int, rle: list[list]):
        self.data[key] = {'width': width, 'height': height, 'rles': rle}

    # Get all data
    def get(self) -> dict:
        return self.data

    # Get the dict of information corresponding to img_path
    def get_entry(self, image_path: str) -> dict:
        if image_path in self.get():
            return self.get()[image_path]
        else:
            print('Error: dataset does not contain image', image_path)
            return {}
        
    # Get all RLE masks for a given image
    def get_rles(self, img_path: str) -> list[list]:
        return [rle for rle in self.get_entry(img_path)['rles']]
    
    # Get all numpy masks for a given image
    def get_masks_np(self, img_path: str) -> list[np.array]:
        rles = self.get_rles(img_path)
        return [mask.decode(rle) for rle in rles]

    # Get available images in the dataset
    def indexes(self) -> list[str]:
        return [img_name for img_name in self.data]

    # Return the number of images in the dataset
    def size(self) -> int:
        return len(self.data)
    
    # Return the number of masks for a given image
    def num_masks(self, index: str) -> int:
        return len(self.get_entry(index)['rles'])

    def display_entry(self, img_path: str) -> None:
        print('Image:', img_path)
        print('Width:', self.get_entry(img_path)['width'])
        print('Height', self.get_entry(img_path)['height'])
        print('Number of Masks:', self.num_masks(img_path))

    # Display information about the dataset.
    def display_all(self):
        output = ''
        for imgPath in self.data:
            output += 'Image: ' + str(imgPath) + '\n'
            output += 'Original width: ' + repr(self.data[imgPath]['width']) + '\n'
            output += 'Original height: ' + repr(self.data[imgPath]['height']) + '\n'
            output += 'Number of masks: ' + repr(len(self.data[imgPath]['rles'])) + '\n'

        print(output)

dataset = Dataset('./dataset')

masks = dataset.get_masks_np('/home/gav/GitHub/TeamRAT/dataset/imgs/3.jpg')

for i in masks:
    cv.imshow('mask', i * 255)
    cv.waitKey(0)

cv.destroyAllWindows()