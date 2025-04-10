# view.py
# Gavin Haynes
# View dataset images along with their COCO annotations.

import os
from sys import argv
import json
import cv2 as cv
import numpy as np
from pathlib import Path
from pycocotools import mask as pycmask

def main():
    assert len(argv) == 2, "Usage: python view.py <path_to_dataset_folder>"