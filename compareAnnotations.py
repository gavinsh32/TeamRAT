import os
import tkinter as tkinter
import cv2 as cv
from tkinter import filedialog as fd

welcome_banner = '''
Welcome to compareAnnotations tool

This tool is designed to show original images along with the annotated ones
to confirm accuracy. 

Your dataset should be organized as following:

dataset1
    /images
    /annotations

Press any key to get started.
'''

initial_dir = '~'

print(welcome_banner)
_ = input()

print('First, select your dataset folder:')
dataset_path = fd.askdirectory(initialdir=initial_dir,
                               title='Select Dataset')
print(dataset_path)

images_path = os.path.join(dataset_path, 'images')
annotations_path = os.path.join(dataset_path, 'annotations')

