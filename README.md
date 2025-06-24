Here's the updated README with the usage commands for `view.py` and `convert.py`:

-----

# Project RAT

**Reconstruction & Analysis of Tendons**

This repository serves as the working environment for the 2025 Capstone Senior Design Project, "R.A.T." Its primary purpose is to house a collection of Python scripts designed to prepare image segmentation datasets for the precise analysis and reconstruction of muscle fascicles (tendon fibers) from micro-CT scans of rat tails.

The toolkit streamlines the data preparation workflow, providing functionalities to:

  * **Convert annotations** generated from Label Studio into a compatible format.
  * **Augment datasets** to enhance their size, diversity, and robustness for machine learning model training.
  * **Visually verify data** at each stage of the preparation process to ensure quality and accuracy.
  * **Train** the Segment Anything Model (SAM) on your prepared datasets.

## Setup

### Dependencies

This project requires **Python 3.10 or newer** and the following core Python packages:

  * `opencv-python`
  * `numpy`
  * `albumentations`
  * `label-studio-converter`

### Installation

Follow these steps to set up the project environment:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/gavinsh32/TeamRAT.git
    cd TeamRAT
    ```

2.  **Create and activate a Python virtual environment** (highly recommended for dependency management):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `.\venv\Scripts\activate`
    ```

3.  **Install the required packages:**

    ```bash
    pip install opencv-python numpy albumentations label-studio-converter
    ```

## Data Preparation Workflow

The data preparation pipeline is orchestrated through three main Python scripts: `convert.py`, `augment.py`, and `view.py`. A `train.py` script is then used to fine-tune the SAM model.

### 1\. Recommended Directory Structure

To ensure a smooth workflow, please organize your raw data and annotations as follows:

```
project_root/
└── dataset/
    ├── imgs/
    │   ├── img0.jpg
    │   ├── img1.jpg
    │   └── ... (etc. - all raw image files)
    └── label-studio-labels.json  (The exported JSON file from Label Studio)
```

### 2\. Usage

Each script plays a distinct role in the data processing pipeline:

  * **`convert.py`**

      * **Purpose:** Decodes segmentation masks from your `label-studio-labels.json` file. It converts these annotations into `uint8` NumPy arrays (which represent the pixel masks) and saves them as `.npy` files. This step standardizes your labels for further processing.
      * **Usage:** `python convert.py <imgs_folder_path> <labels_json_path> <output_json_path>`

  * **`view.py`**

      * **Purpose:** Provides a utility to visually inspect your training images and their corresponding masks. This is crucial for verifying that the `convert.py` script worked correctly and that your annotations are accurately represented.
      * **Usage:** `python view.py <imgs_folder_path> <masks_folder_path>`

  * **`augment.py`**

      * **Purpose:** Enhances your training images and masks through various augmentation techniques. It applies transformations such as modifying gamma, rotation, and blur. This script also scales the total volume of your dataset by a constant factor, increasing the dataset's size and diversity to improve model robustness.

  * **`train.py`**

      * **Purpose:** Initiates the training process for the Segment Anything Model (SAM) using your prepared and augmented dataset. This script handles the fine-tuning of the model to automate the fascicle extraction task.

-----