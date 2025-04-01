# rename.py
# Gavin Haynes
# Take png masks exported from Label Studio
# and rename so they match input image.

import glob
import os
import shutil

mask_folder = './masks'

# Create output folder for images
if not os.path.exists(mask_folder):
    os.mkdir(mask_folder)

image = 1
while True:
    ls_name = 'task-' + repr(image) + '-*.png'
    masks = glob.glob(ls_name)
    
    if len(masks) < 1:
        break
    
    counter = 0
    for mask in masks:
        dest = mask_folder + '/image-' + repr(image) + '-mask-' + repr(counter) + '.png'
        shutil.copy(mask, dest)
        counter += 1

    image += 1