# dataset.py
# Gavin Haynes
# Read in Label Studio Annotations, fetch labels, and check your dataset.

import json                     # Reading annotations in
import numpy as np              # Mask and image data
import os                       # System operations
from pathlib import Path        # Path manipulation
import cv2 as cv
import labelconverter as lc     # See labelconvert.py, I did not write this
import albumentations as A

class Dataset:

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

                # Create a new entry with essential information   
                self.data[img_path] = {
                    'width': results[0]['original_width'],
                    'height': results[0]['original_height'],
                    'rle': [result['value']['rle'] for result in results]
                }

    def entries(self):
        return self.data.keys()

    # Get all data
    def get(self):
        """
        Get the dict containing dataset information.

        Returns: dict
        """
        return self.data

    # Get all the numpy masks for an image
    def getMasks(self, img_path: str):
        entry = self.get()[img_path]
        width, height, rles = entry['width'], entry['height'], entry['rle']
        return [lc.rle_to_mask(rle, width, height) for rle in rles]
    
    def display(self):
        output = ''
        for imgPath in self.data:
            output += 'Image: ' + str(imgPath) + '\n'
            output += 'Original width: ' + repr(self.data[imgPath]['width']) + '\n'
            output += 'Original height: ' + repr(self.data[imgPath]['height']) + '\n'
            output += 'Number of masks: ' + repr(len(self.data[imgPath]['rle'])) + '\n'

        print(output)

dataset = Dataset('./dataset')

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

output_imgs = []
output_masks = []

# Augment images
for img_path in dataset.get():
    input_masks = dataset.getMasks(img_path)
    input_img = cv.imread(img_path)
    