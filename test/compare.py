# compare.py
# Gavin Haynes
# View input images along with annotations.


from typing import List
import numpy as np
import cv2 as cv
import json
import labelconverter as lb

with open('./data.json') as data_file:
    data = json.load(data_file)

    for task in data:
        print('Viewing labels for task', task['id'])
        for annotation in task['annotations'][0]['result']:
            rle = annotation['value']['rle']
            width = annotation['original_width']
            height = annotation['original_height']
            mask = lb.rle_to_mask(rle, height, width)
            cv.imshow('Result', mask)
            cv.waitKey(0)