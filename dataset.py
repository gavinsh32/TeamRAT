# dataset.py
# Gavin Haynes
# Read in Label Studio Annotations, fetch labels, and check your dataset.

import json                     # Reading annotations in
import numpy as np              # Mask and image data
import os                       # System operations
from pathlib import Path        # Path manipulation
import cv2 as cv
import pycocotools.mask as mask # Mask re-encoding
import labelconverter as lc     # See labelconvert.py, I did not write this

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
    def get_all(self) -> dict:
        return self.data
        
    # Get all RLE masks for a given image
    def get_rles(self, img_path: str) -> list[list]:
        return [rle for rle in self.get_all()[img_path]['rles']]
    
    # Get all numpy masks for a given image
    def get_masks(self, img_path: str) -> list[np.array]:
        rles = self.get_rles(img_path)
        return [mask.decode(rle) for rle in rles]

    # Get available images in the dataset
    def indexes(self) -> list[str]:
        return [img_name for img_name in self.data]
    
    def show(self, img_path: str, option=0):
        """
        Show the image and its masks.
        Args:
            img_path: path to the image
            option
        """
        masks = self.get_masks(img_path)
        img = cv.imread(img_path)

        for contours in self.get_contours(img_path):
            cv.drawContours(img, contours, -1, (0, 0, 255), 2)

        name = img_path.split('/')[-1] + ', # of Labels: ' + str(len(masks))
        cv.imshow(name, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Display information about the dataset.
    def display_all(self):
        output = ''
        for imgPath in self.data:
            output += 'Image: ' + str(imgPath) + '\n'
            output += 'Original width: ' + repr(self.data[imgPath]['width']) + '\n'
            output += 'Original height: ' + repr(self.data[imgPath]['height']) + '\n'
            output += 'Number of masks: ' + repr(len(self.data[imgPath]['rles'])) + '\n'

        print(output)

    def get_bounding_rects(self, img_path: str):
        bboxes = []
        for contour in self.get_contours(img_path):
                x, y, w, h = cv.boundingRect(contour)
                bboxes.append((x, y, w, h))
        return bboxes

    def get_contours(self, img_path: str):
        all_contours = []
        for mask in self.get_masks(img_path):
            mask *= 255
            contours, heirarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            all_contours.append(contours)
        return all_contours

dataset = Dataset('./dataset')

img_path = dataset.indexes()[0]

img = cv.imread(img_path)

bboxes = dataset.get_bounding_rects(img_path)

for box in bboxes:
    x, y, w, h = box
    cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()