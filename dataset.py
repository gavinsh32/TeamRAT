# dataset.py
# Gavin Haynes
# Read in Label Studio Annotations, fetch labels, and check your dataset.

import json                     # Reading annotations in
import numpy as np              # Mask and image data
import cv2 as cv                # Display results
import os                       # System operations
from pathlib import Path        # Path manipulation
import labelconverter as lc     # See labelconvert.py, I did not write this

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

        # Iterate through each task (set of annotations, 1 per input image)
        for task in data:
            
            # Grab name of the source image that was annotated, and trim
            img_name = task['file_upload'].split('-')[1]
            img_path = str(imgs_folder / img_name)

            # Only read in annotations if there is a matching image
            if os.path.exists(img_path):
                
                # Get all annotation information for an image
                results = task['annotations'][0]['result']

                # Create a new entry with essential information   
                self.data[img_path] = {
                    'width': results[0]['original_width'],
                    'height': results[0]['original_height'],
                    'rle': [result['value']['rle'] for result in results]
                }

    # Get all data
    def get(self):
        """
        Get the dict containing dataset information.

        Returns: dict
        """
        return self.data
    
    def display(self):
        output = ''
        for imgPath in self.data:
            output += 'Image: ' + str(imgPath) + '\n'
            output += 'Original width: ' + repr(self.data[imgPath]['width']) + '\n'
            output += 'Original height: ' + repr(self.data[imgPath]['height']) + '\n'
            output += 'Number of masks: ' + repr(len(self.data[imgPath]['rle'])) + '\n'

        return output