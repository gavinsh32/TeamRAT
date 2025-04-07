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

                # Create a new entry with essential information   
                self.insert(
                    img_path,
                    results[0]['original_width'],
                    results[0]['original_height'],
                    [result['value']['rle'] for result in results]
                )

    # Augment the dataset, scaling by constant factor scale
    def augment(self, scale=5):
        """
        Augment images loaded in to the datset, scaling the volume by a constant factor. Augmented images are appended to current dataset and saved in the output folder.
        Args:
            scale:
            Scale the total number of images by this amount
        Returns: None
        """

        # Transformation Pipeline
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])

        for img_path in self.get():

            input_masks = self.get_masks(img_path)
            input_img = cv.imread(img_path, cv.COLOR_RGB2BGR)

            for _ in range(scale):

                result = transform(image=input_img, masks=input_masks)
                
                output_img = result['image']
                output_masks = result['masks']
                
                ex = mask.encode(np.asfortranarray(output_masks[0]))
                
                print(ex)

    # Insert an image with annotations to internal dataset
    def insert(self, key: str, width: int, height: int, rle: list[list[int]]):
        self.data[key] = {'width': width, 'height': height, 'rle': rle}

    # Get all data
    def get(self) -> dict:
        return self.data

    def get_masks_rle(self, image_path: str) -> list[list]:
        return self.get_entry(image_path)['rle']

    # Get the dict of information corresponding to img_path
    def get_entry(self, image_path: str) -> dict:
        if image_path in self.get():
            return self.get()[image_path]
        else:
            print('Error: dataset does not contain image', image_path)
            return {}
        
    def get_mask_rle(self, index: str, mask_num: int) -> int:
        if mask_num < self.num_masks(index):
            return self.get_entry(index)['rle'][mask_num]
        else:
            print('Error: mask index', mask_num, 'out of range.')
            return -1
    
    # Get all the numpy masks for an image
    def get_masks(self, img_path: str) -> list[np.array]:
        return [self.get_mask(img_path, i) for i in self.num_masks('img_path')]
    
    def get_mask(self, img_path: str, mask_num: int) -> np.array:
        entry = self.get_entry(img_path)
        width = entry['width']
        height = entry['height']
        rle = self.get_mask_rle(img_path, mask_num)
        return lc.rle_to_mask(rle, height, width)

    def indexes(self) -> list[str]:
        return [img_name for img_name in self.data]

    def size(self) -> int:
        return len(self.data)
    
    def num_masks(self, index: str) -> int:
        return len(self.get_entry(index)['rle'])

    def display(self, option: int) -> None:
        pass
    
    def display_all(self, image_path: str):
        pass

    def display(self, image_path: str):
        pass

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
            output += 'Number of masks: ' + repr(len(self.data[imgPath]['rle'])) + '\n'

        print(output)

dataset = Dataset('./dataset')

mask_orig = dataset.get_mask('/home/gav/GitHub/TeamRAT/dataset/imgs/2.jpg', 0)

# cv.imshow('Original Mask', mask_orig)
# cv.waitKey(0)

mask_encoded = mask.encode(np.asfortranarray(mask_orig))
#print(mask_encoded)

mask_decoded = mask.decode(mask_encoded)

mask_decoded = mask_decoded.reshape(mask_decoded.shape[0], mask_decoded.shape[1], 1)

print(mask_decoded.dtype)
print(mask_decoded.shape)
print(np.unique(mask_decoded))

cv.imshow('Mask', mask_decoded * 255)
cv.waitKey(0)
cv.destroyAllWindows()