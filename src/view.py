# view.py
# Gavin Haynes
# Display training images and masks.

import sys
import cv2 as cv
import numpy as np
from pathlib import Path

def main():
    assert len(sys.argv) == 3, 'Usage: python view.py <imgs_folder_path> <masks_folder_path>'

    # Resolve the paths to the images and masks directories.
    imgs_dir: Path = Path(sys.argv[1]).resolve()
    masks_dir: Path = Path(sys.argv[2]).resolve()
    
    assert imgs_dir.is_dir(), f'Images directory does not exist: {imgs_dir}'
    assert masks_dir.is_dir(), f'Masks directory does not exist: {masks_dir}'

    # Fetch all masks for a given image.
    for img_path in imgs_dir.glob('*.jpg'):
        img: np.ndarray = cv.imread(str(img_path))
        mask: np.ndarray = np.load(masks_dir / f'{img_path.stem}.npy')
        mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        output: np.ndarray = cv.addWeighted(img, 1.0, mask, 0.3, 0)
        
        cv.imshow('View Training Data', output)
        cv.waitKey(0)

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()