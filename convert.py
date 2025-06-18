# convert.py
# Gavin Haynes
# Convert Label-Studio RLE annotations to COCO format for use with training SAM.

import os
import sys
import json
import cv2 as cv
import numpy as np
from pathlib import Path
from label_studio_converter.brush import decode_rle
from pycocotools import mask as pycmask

def main():
    assert len(sys.argv) == 4, "Usage: python convert.py <imgs_folder_path> <labels_json_path> <output_json_path>"
    
    imgs_folder_path: Path = Path(sys.argv[1]).resolve()
    input_labels_path: Path = Path(sys.argv[2]).resolve()
    output_dir: Path = Path(sys.argv[3]).resolve()

    os.mkdir(output_dir)
    
    dataset: dict = {}

    with open(input_labels_path, 'r') as data_file:
        data = json.load(data_file)
        
        for task in data:   
            # Grab name of the source image that was annotated, and trim
            img_name = task['file_upload'].split('-')[1]
            
            # Import masks only if there's a matching image.
            if os.path.exists(imgs_folder_path / img_name):
                print('Processing:', img_name)
                masks = []
                
                for i, result in enumerate(task['annotations'][0]['result']):
                    height = result['original_height']
                    width = result['original_width']
                    mask: np.ndarray = decode_rle(result['value']['rle']).astype(np.uint8)
                    mask[mask > 0] = 255
                    mask = np.reshape(mask, (height, width, 4))
                    mask = mask[:, :, 3]
                    mask_name = img_name.split('.')[0] + f'-{i}.npy'
                    np.save(output_dir / mask_name, mask)

if __name__ == "__main__":
    main()