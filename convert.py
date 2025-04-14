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
                coco_dataset.append(ls_to_coco_entry(img_path, results))
    
    output_json_path = dataset_folder_path / 'converted.json'

    with open(output_json_path, 'w') as output_file:
        json.dump(coco_dataset, output_file, indent=4)

def ls_to_coco_entry(img_path: str, results: list):
    """
    Convert a LS entry to COCO format.
    Args:
        img_path (str): Path to the image.
        results (list): List of segmentation results in LS format.
    Returns:
        dict: COCO dataset entry.
    """
    
    height = results[0]['original_height']
    width = results[0]['original_width']
    
    image_info = {
        'image_id': img_path,
        'width': width,
        'height': height,
        'file_name': img_path
    }

    # Create annotation entries for each mask in the results

    annotations = []
    
    for id, annotation in enumerate(results):

        coco_rle = convert_rle(annotation)
        json_rle = coco_rle.copy()
        json_rle['counts'] = json_rle['counts'].decode('utf-8')

        annotations.append({
            'id': id,
            'segmentation': json_rle,
            'bbox': pycmask.toBbox(coco_rle).tolist(),
            'area': int(pycmask.area(coco_rle)),
            'predicted_iou': 0.0,
            'stability_score': 0.0,
            'crop_box': pycmask.toBbox(coco_rle).tolist(),
            'point_coords': []
        })

    return {'image': image_info, 'annotations': annotations}

def convert_rle(label: list) -> dict:
    """
    Convert an RLE mask to COCO format.
    Args:
        label (list): Label-Studio RLE string).
    Returns:
        dict: COCO format RLE mask. 
    """
    
    mask = rle_to_mask(
        label['value']['rle'],
        label['original_height'],
        label['original_width']
    )

    mask[mask > 0] = 1
    mask = np.asfortranarray(mask, dtype=np.uint8)
       
    return pycmask.encode(mask)
        
if __name__ == "__main__":
    main()
    cv.destroyAllWindows()