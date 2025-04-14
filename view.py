# view.py
# Gavin Haynes
# View dataset images along with their COCO annotations.

import os
from sys import argv
import json
import cv2 as cv
import numpy as np
import dataset
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
        image_path, height, width, masks = dataset.get_img_data(entry)
        
        cv.imshow(image_path, dataset.draw_contours(image_path, masks))
        cv.waitKey(0)

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()