# augment.py
# Gavin Haynes

import cv2 as cv
import sys
import numpy as np
import albumentations as A
from dataset import Dataset
from pathlib import Path

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.GaussNoise(p=0.25),
])

def main():
    assert len(sys.argv) == 3, "Usage: python augment.py <path_to_img_folder> <path_to_json_file>"

    img_folder_path = Path(sys.argv[1]).resolve()
    data_path = Path(sys.argv[2]).resolve()

    dataset = Dataset(img_folder_path, data_path)

    for entry in dataset.get_all():
        img_path, height, width, masks = dataset.get_img_data(entry)
        
        img = cv.imread(str(img_folder_path / img_path))

        results = transform(image=img, masks=masks)

        result_img = results['image']
        result_masks = results['masks']

        cv.imshow('Original:', dataset.draw_contours(img, masks))

        cv.imshow('Result:', dataset.draw_contours(result_img, result_masks))
        cv.waitKey(0)

if __name__ == "__main__":
    main()