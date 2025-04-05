# dataset.py
# Gavin Haynes
# Read in Label Studio Annotations, fetch labels, and check your dataset.

import json
import labelconverter as lc
import numpy as np
import cv2 as cv
from pathlib import Path

class Dataset:
    def __init__(self, images_path: Path, annotations_path: Path):
        self.images_path = images_path.resolve()
        self.annotations_path = annotations_path.resolve()
        print(self.images_path)
        print(self.annotations_path)

    def check(self, imgs_path: Path):
        pass
    
    # Load JSON File and store the dict
    def loadJSON(self, json_path: Path) -> bool:
        return True

cwd = Path.cwd().absolute()
imgs_path = cwd.joinpath('imgs')

print(cwd)
print(imgs_path)

# dataset = Dataset('./imgs/')