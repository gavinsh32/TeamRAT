import os
import tkinter as tkinter
from tkinter import filedialog as fd

images_path = ''
annotations_path = ''
welcome_banner = '''
Welcome to compareAnnotations tool
'''

print(welcome_banner)

images_path = fd.askdirectory()
annotations_path = fd.askdirectory(title='Select Annotations Folder')