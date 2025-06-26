# augment.py
# Apply pre-determined augmentations to training data, also scaling the volume by a factor.

import sys
import albumentations as A
import cv2 as cv
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ratdataset import RatDataset

def main():
    assert len(sys.argv) == 5, 'Usage: python augment.py <imgs_folder_path> <masks_folder_path> <output_folder_path> <scale_factor>'

    # Resolve paths using pathlib
    imgs_folder_dir: Path = Path(sys.argv[1]).resolve()
    masks_folder_dir: Path = Path(sys.argv[2]).resolve()
    output_folder_dir: Path = Path(sys.argv[3]).resolve()
    scale_factor: int = int(sys.argv[4])

    # Create output directories safely
    output_images_dir: Path = output_folder_dir / 'imgs'
    output_masks_dir: Path = output_folder_dir / 'masks'
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    # Define the augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
    ])

    # Use RatDataset to load data
    print(f"Loading data from {imgs_folder_dir} and {masks_folder_dir}...")
    dataset = RatDataset(imgs_dir=imgs_folder_dir, masks_dir=masks_folder_dir)
    print(f"Found {len(dataset)} images. Starting augmentation to scale by a factor of {scale_factor}.")

    # Iterate 'scale_factor' times to generate multiple augmentations per original image
    for i in range(scale_factor):
        # Iterate through each image-mask pair in the dataset
        for idx in tqdm(range(len(dataset)), desc=f"Augmentation set {i+1}/{scale_factor}"):
            img_path = dataset.img_paths[idx]
            img, mask = dataset[idx]

            # Apply augmentation
            augmented = transform(image=img, mask=mask)
            # Create new, unique filenames for the augmented data
            new_img_name = f"{img_path.stem}_aug_{i}.jpg"
            new_mask_name = f"{img_path.stem}_aug_{i}.npy"
            # Save augmented files
            cv.imwrite(str(output_images_dir / new_img_name), augmented['image'])
            np.save(output_masks_dir / new_mask_name, augmented['mask'])

    print(f"\nAugmentation complete. Generated {len(dataset) * scale_factor} new image/mask pairs.")
    print(f"Output images: {output_images_dir}")
    print(f"Output masks: {output_masks_dir}")

if __name__ == "__main__":
    main()