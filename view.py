# view.py
# Gavin Haynes
# View dataset images along with their COCO annotations.

import os
from sys import argv
import json
import cv2 as cv
import numpy as np
from dataset import Dataset
from pathlib import Path
from pycocotools import mask as pycmask

def main():
    assert len(argv) == 2, "Usage: python view.py <path_to_dataset_folder>"

    dataset_path = Path(argv[1]).resolve()
    data_path = dataset_path / 'labels.json'
    imgs_folder_path = dataset_path / 'imgs'

    data = []

    with open(data_path) as data_file:
        data = json.load(data_file)

    assert len(data) > 0, 'ERROR: No data found in ' + str(data_path)

    dataset = Dataset(imgs_folder_path, data_path)

    for i, entry in enumerate(dataset.get_all()):
        image_path, height, width, masks = dataset.get_img_data(entry)
        
        # result = np.zeros((height, width, 3), dtype=np.uint8)
        
        result = dataset.draw_contours(cv.imread(image_path), masks)

        cv.imshow(f'masks/mask{i}.png', result)
        cv.waitKey(0)

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()