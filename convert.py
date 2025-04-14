# convert.py
# Gavin Haynes
# Import Label Studio segmentation annotations and export to COCO RLE format.

import os
import sys
import json
import cv2 as cv
import numpy as np
import dataset as d
from labelconverter import rle_to_mask
from pathlib import Path
from pycocotools import mask as pycmask


def main():
    
    assert len(sys.argv) == 2, "Usage: python convert.py <path_to_dataset_folder>"
    
    dataset_folder_path = Path(sys.argv[1]).resolve()
    imgs_folder_path = dataset_folder_path / 'imgs'
    
    coco_dataset = []

    with open(dataset_folder_path / 'ls-labels.json') as data_file:
        
        data = json.load(data_file)
        
        # Loop through each task, one set of annotations per image
        for task in data:   

            # Grab name of the source image that was annotated, and trim
            img_name = task['file_upload'].split('-')[1]
            img_path = str(imgs_folder_path / img_name)
            
            # Only read in annotations if there is a matching image
            if os.path.exists(img_path):

                # List of masks corresponding to an image
                results = task['annotations'][0]['result']
                coco_dataset.append(d.ls_to_coco_entry(img_path, results))
    
    output_json_path = dataset_folder_path / 'converted.json'

    with open(output_json_path, 'w') as output_file:
        json.dump(coco_dataset, output_file, indent=4)
        
if __name__ == "__main__":
    main()
    cv.destroyAllWindows()