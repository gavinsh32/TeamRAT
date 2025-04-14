# augment.py
# Gavin Haynes

import cv2 as cv
import numpy as np
import albumations as A

def main():
    # Load an image
    image = cv.imread('image.jpg')

    # Define the augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Random90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=45, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5)
    ])