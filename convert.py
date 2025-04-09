# convert.py
# Gavin Haynes
# Import Label Studio segmentation annotations and export to COCO RLE format.

import os
import sys
import json
import cv2 as cv
import numpy as np
from labelconverter import rle_to_mask
from pathlib import Path
from pycocotools import mask as pycmask

def main():
    
    assert len(sys.argv) == 2, "Usage: python convert.py <dataset_folder_path>"
    
    dataset_folder_path = Path(sys.argv[1]).resolve()
    imgs_folder_path = dataset_folder_path / 'imgs'
    parent_folder_path = dataset_folder_path.parent

    coco_dataset = []

    with open(dataset_folder_path / 'labels.json') as data_file:
        
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
                coco_dataset.append(make_entry(img_path, results))


    print(len(coco_dataset))

    output = open(parent_folder_path / 'coco_dataset.json', 'w')
    json.dump(coco_dataset, output, indent=4)

def make_entry(img_path: str, results: list):

    height = results[0]['original_height']
    width = results[0]['original_width']
    
    image_info = {
        'image_id': img_path,
        'width': width,
        'height': height,
        'file_name': img_path
    }

    annotations = []

    for i in range(len(results)):
        
        rle = rle_to_mask(results[i]['value']['rle'], height, width)
        rle[rle > 0] = 1
        rle = np.asfortranarray(rle, dtype=np.uint8)
        res = pycmask.encode(rle)
        
        annotations.append({
            'id': i,
            'segmentation': res,
            'bbox': pycmask.toBbox(res),
            'area': pycmask.area(res),
            'predicted_iou': 0.0,
            'stability_score': 0.0,
            'crop_box': pycmask.toBbox(res),
            'point_coords': []
        })
    
    return {'image': image_info, 'annotations': annotations}

# Convert ls rle to coco rle
def convert_rles(name: list, results: list):
    rles = []
    for label in results:
        mask = rle_to_mask(
            label['value']['rle'],
            label['original_height'],
            label['original_width']
        )
        mask[mask > 0] = 1
        mask = np.asfortranarray(mask, dtype=np.uint8)
        rles.append(pycmask.encode(mask))
    return rles

        
if __name__ == "__main__":
    main()
    cv.destroyAllWindows()