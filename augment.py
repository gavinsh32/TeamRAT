# augment.py
# Apply pre-determined augmentations to training data, also scaling the volume by a factor.

import os
import sys
import albumentations as A
import cv2 as cv
import numpy as np
from pathlib import Path

def main():
    assert len(sys.argv) == 5, 'Usage: python augment.py <imgs_folder_path> <masks_folder_path> <output_folder_path> <scale_factor>'

    # Check paths for images, masks, and output directory.
    imgs_folder_dir: Path = Path(sys.argv[1]).resolve()
    masks_folder_dir: Path = Path(sys.argv[2]).resolve()
    output_folder_dir: Path = Path(sys.argv[3])
    scale_factor: int = int(sys.argv[4])

    if not os.path.exists(output_folder_dir):
        os.mkdir(output_folder_dir)
        output_folder_dir.resolve()

    output_images_dir: Path = output_folder_dir / 'img'
    output_masks_dir: Path = output_folder_dir / 'masks'

    os.mkdir(output_images_dir)
    os.mkdir(output_masks_dir)

    # Define the series of augmentations to apply.
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
        A.Blur(blur_limit=2, p=0.2),
    ])

    # Iterate through all input images.
    for img_counter, img_path in enumerate(imgs_folder_dir.glob('*.jpg')):
        img: np.ndarray = cv.imread(str(img_path))
        masks: list[np.ndarray] = []
        
        # Load numpy masks and store in masks list.
        for i, mask_path in enumerate(masks_folder_dir.glob(f'{img_path.stem}-*.npy')):
            mask: np.ndarray = np.load(mask_path)
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            masks.append(mask)

        # Apply augmentations many times for each set of images 
        # and masks, scaling the dataset by a constant factor.
        for i in range(scale_factor):
            results = transform(image=img, masks=masks)
            transformed_img: np.ndarray = results['image']
            transformed_masks: list[np.ndarray] = results['masks']
            img_path: Path = output_images_dir / f'{img_counter}-i.jpg'
            
            # Save the transformed masks.
            for transformed_mask in transformed_masks:
                mask_path: Path = output_masks_dir / f'{img_counter}-i-{transformed_masks.index(transformed_mask)}.npy'
                cv.imwrite(str(img_path), transformed_img)
                np.save(mask_path, transformed_mask)

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()