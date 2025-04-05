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

    # Read in dataset and load valid images and annotations
    def __init__(self, dataset_folder_path: str):

        dataset_folder = Path(dataset_folder_path).resolve()
        imgs_folder = dataset_folder / 'imgs'

        # Read in essential data from labels
        labels_file = dataset_folder / 'labels.json'
        with open(labels_file) as data_file:

            self.data = {}

            data = json.load(data_file)

            # Grab image path from json
            for task in data:
                # Grab name of matching file, trim, and append to folder path
                img_path = imgs_folder / task['file_upload'].split('-')[1]

                if os.path.exists(img_path):

                    self.data[img_path] = {}
                    results = task['annotations'][0]['result']

                    # Copy width and height from first annotation
                    self.data[img_path]['width'] = results[0]['original_width']
                    self.data[img_path]['height'] = results[0]['original_height']

                    # All rle masks for the img
                    self.data[img_path]['rles'] = [result['value']['rle'] for result in results]

    # Get all data
    def get(self):
        return self.data
    
    # Display all items in the dataset along with the number of annotations
    def __str__(self):
        output = ''
        for imgPath in self.data:
            output += 'Image: ' + str(imgPath) + '\n'
            output += 'Original width: ' + repr(self.data[imgPath]['width']) + '\n'
            output += 'Original height: ' + repr(self.data[imgPath]['height']) + '\n'
            output += 'Number of masks: ' + repr(len(self.data[imgPath]['rles'])) + '\n'

        return output

dataset = Dataset('./dataset')

print(dataset)

data = dataset.get()

for key in data:
    img_path = repr(key)
    print('Reading labels for', img_path)
    width = data[key]['width']
    height = data[key]['height']
    for rle in data[key]['rles']:
        mask = lc.rle_to_mask(rle, width, height)
        cv.imshow('Result', mask)
        cv.waitKey(0)