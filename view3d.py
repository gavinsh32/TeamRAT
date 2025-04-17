import plotly.graph_objects as go
import numpy as np
import cv2 as cv
import sys
from pathlib import Path
from dataset import Dataset

assert len(sys.argv) == 2, "Usage: python view3d.py <path_to_dataset_folder>"

dataset_path = Path(sys.argv[1]).resolve()
imgs_path = dataset_path / 'imgs'
data_path = dataset_path / 'labels.json'

data = Dataset(imgs_path, data_path)

masks = []

for entry in data.get_all():
    image_path, height, width, labels = data.get_img_data(entry)

    bg = np.zeros((height, width), dtype=np.uint8)

    for label in labels:
        bg += label[label > 0]

    cv.imshow('Image', bg)
    cv.waitKey(0)


print(len(masks))