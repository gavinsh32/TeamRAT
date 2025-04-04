# rename.py
# Gavin Haynes
# Take exported png masks along with JSON and rename all files consistently

# Open JSON
# loop
#   grab next task
#   get name of input
#   get name of exported masks
#   rename and save renamed annotations

import os
import shutil
import cv2
import json
import glob

# Create output folder for renamed masks
folderCounter = 0
output_folder_name = 'masks'
while os.path.exists('./masks' + repr(folderCounter)):
    folderCounter += 1
os.mkdir(output_folder_name + repr(folderCounter))
output_folder_path = output_folder_name + repr(folderCounter)

# Open JSON
with open('data.json') as data_file_path:
    
    # Open JSON file
    data_file = json.load(data_file_path)
    
    # Iterate through each task (set of annotations)
    for i, task in enumerate(data_file):

        # Grab name of file that corresponds to task and trim
        input_image_name = task['file_upload']
        input_image_name = input_image_name.split('-')[1]

        print('Loading task', task['id'])
        print('Searching for input image:', input_image_name + '...')

        # Found matching input image
        if os.path.exists('imgs/'+input_image_name):
            
            print('Found', input_image_name)
            print('Searching for masks...')
            
            # Grab masks and rename to match input image
            mask_counter = 0
            for mask in glob.glob('masks/task-' + repr(task['id']) + '-*'):
                new_name = output_folder_path + '/' + input_image_name.split('.')[0] + '-mask' + repr(mask_counter) + '.png'
                print('Renaming', mask, 'to', new_name)
                shutil.copy(mask, new_name)
                mask_counter += 1
        else:
            print('Couldn\'t find imgs/' + input_image_name)

        print()