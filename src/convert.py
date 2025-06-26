# convert.py | Gavin Haynes

# Convert Label-Studio RLE annotations to numpy masks and save. Label Studio uses a custom RLE format for masks, which we need to convert to a format which is human-readable and easily modifiable. Masks are decoded and are subsequently saved as uint8 numpy arrays, the standard data format for OpenCV and image processing.

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
    output_masks_dir: Path = output_dir / 'masks'
    
    assert imgs_folder_path.is_dir(), f"Images directory does not exist: {imgs_folder_path}"
    assert input_labels_path.is_file(), f"Labels JSON file does not exist: {input_labels_path}"
    assert output_dir.is_dir() or not output_dir.exists(), f"Output directory does not exist: {output_dir}"

    os.mkdir(output_dir)
    os.mkdir(output_masks_dir)

    # Begin reading LS annotations.
    with open(input_labels_path, 'r') as data_file:
        data = json.load(data_file)
        
        # Iterate through all sets of annotations (one image has many masks).
        for task in data:   
            # Grab name of the source image that was annotated, and trim
            img_name = task['file_upload'].split('-')[1]
            
            # Import masks only if there's a matching image.
            if os.path.exists(imgs_folder_path / img_name):

                height = task['annotations'][0]['result'][0]['original_height']
                width = task['annotations'][0]['result'][0]['original_width']
                mask = np.zeros((height, width), dtype=np.uint8)
                
                # Iterate through all masks ('results') for an image.
                for i, result in enumerate(task['annotations'][0]['result']):

                    # Convert LS-style dict {'width': 100, 'height': 100, 'rle': '...'} to
                    # COCO-style 1D ndarray.
                    label: np.ndarray = decode_rle(result['value']['rle']).astype(np.uint8)

                    # Convert mask to human-readable format. 
                    # COCO expects masks to be in the format (b, g, r, a).
                    label = np.reshape(label, (height, width, 4))

                    # Convert to cv2 friendly format and normalize, dropping alpha channel.
                    label = label[:, :, 3]
                    label[label > 0] = 255    # move all non-white pizels to black.
                    mask |= label

                # Export mask to numpy format and name accordingly.
                mask_name = img_name.split('.')[0] + '.npy'
                np.save(output_masks_dir / mask_name, mask)

if __name__ == "__main__":
    main()