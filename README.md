# Project RAT
Working repository for 2025 Capstone Senior Design Project: R.A.T. (Reconstruction & Analysis of Tendons).

This repository contains various tools for deploying a fine-tuned SAM model, including dataset validation, augmentation, and training.

## Setup

### Dataset Folder

These modules are designed to work with Label Studio annotations: segmentation with masks (in RLE format). Export labels as JSON and organize input images as shown below.

* dataset/
    
    * imgs/

        * img0.jpg
        * img1.jpg
        * ...
        * img99.jpg

    * data.json 

### Dependencies
* python 3.10.12
* pip 22.0.2
* opencv-python 4.11.0.86
* numpy 1.26.4
* pycocotools 2.0.8
* albumations 2.0.5

### Installation
Get the repository:
```
git clone https://github.com/gavinsh32/TeamRAT.git
```
Install dependencies
```
cd <path/to/TeamRAT> && pip install -r requirements.txt
```

## Usage