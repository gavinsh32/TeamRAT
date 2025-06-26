# train.py | Gavin Haynes
#
# Fine-tune a Segment Anything Model (SAM) to segment fascicles from micro-CT scans of rat tails.
# This script is a refactored, command-line-driven version of the original Colab notebook.

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
from statistics import mean

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as pycmask
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm

from ratdataset import RatDataset

# Helper functions from the notebook
def make_point_grid(height: int, width: int, n: int, padding: int):
    """Return a point prompt grid of n x n points."""
    start_y, start_x = padding, padding
    end_y, end_x = height - padding, width - padding
    x_step = (end_x - start_x) // n
    y_step = (end_y - start_y) // n
    x_points = [x for x in range(start_x, end_x + 1, x_step)]
    y_points = [y for y in range(start_y, end_y + 1, y_step)]
    points = []
    for y in y_points:
        for x in x_points:
            points.append([y, x])
    return points

def preprocess_images_for_sam(image_keys: list, image_dir: Path, sam_model, device: str):
    """
    Preprocesses images for SAM, applying transformations and creating tensors.
    """
    print("Preprocessing images for SAM...")
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    transformed_data = defaultdict(dict)

    for k in tqdm(image_keys, desc="Preprocessing images"):
        img_path = image_dir / k
        if not img_path.exists():
            print(f"Warning: Image file not found, skipping: {img_path}")
            continue

        img = cv.imread(str(img_path))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        input_img = transform.apply_image(img)
        input_img_torch = torch.as_tensor(input_img, device=device)
        input_img_torch = input_img_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        transformed_data[k]['image'] = sam_model.preprocess(input_img_torch)
        transformed_data[k]['input_size'] = tuple(input_img_torch.shape[-2:])
        transformed_data[k]['original_image_size'] = img.shape[:2]

    print(f"Preprocessed {len(transformed_data)} images.")
    return transformed_data

def run_training_loop(sam_model, optimizer, loss_fn, dataset: RatDataset, transformed_data, epochs, device):
    """
    Executes the main training loop for fine-tuning the SAM mask decoder.
    """
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    
    # Define the point grid prompt
    num_points, padding = 19, 20
    # Assuming all images are the same size for the grid for simplicity
    height, width = 660, 660 
    full_grid_points = np.array(make_point_grid(height, width, num_points, padding))

    all_epoch_losses = []
    print(f"\nStarting training for {epochs} epochs...")

    for epoch in range(epochs):
        sam_model.train()
        epoch_losses = []
        
        # Iterate directly over the dataset
        for k, (_, gt_mask_np) in tqdm(dataset, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            # k is the filename, _ is the raw image (not needed), gt_mask_np is the mask
            
            if k not in transformed_data:
                print(f"Warning: Skipping {k} as it was not found in preprocessed data.")
                continue

            input_image = transformed_data[k]['image'].to(device)
            input_size = transformed_data[k]['input_size']
            original_image_size = transformed_data[k]['original_image_size']

            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)
                
                points_coords_transformed = transform.apply_coords(full_grid_points, original_image_size)
                points_torch = torch.as_tensor(points_coords_transformed, dtype=torch.float, device=device).unsqueeze(0)
                labels_torch = torch.ones(points_torch.shape[1], dtype=torch.int, device=device).unsqueeze(0)

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=(points_torch, labels_torch),
                    boxes=None,
                    masks=None,
                )

            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled_masks = sam_model.postprocess_masks(
                low_res_masks,
                input_size=input_size,
                original_size=original_image_size
            ).to(device)

            gt_mask_torch = torch.from_numpy(gt_mask_np).unsqueeze(0).unsqueeze(0).to(device)
            
            # Ensure GT mask and predicted mask have the same spatial dimensionss
            target_h, target_w = upscaled_masks.shape[-2:]
            if gt_mask_torch.shape[-2:] != (target_h, target_w):
                 gt_mask_torch = F.interpolate(gt_mask_torch,
                                               size=(target_h, target_w),
                                               mode='nearest')

            loss = loss_fn(upscaled_masks, gt_mask_torch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        all_epoch_losses.append(epoch_losses)
        mean_loss = mean(epoch_losses) if epoch_losses else 0
        print(f'Epoch {epoch+1}/{epochs} - Mean Loss: {mean_loss:.6f}')

    return all_epoch_losses

def plot_loss(losses, output_dir: Path):
    """Plots and saves the training loss curve."""
    mean_losses = [mean(x) if x else 0 for x in losses]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(mean_losses) + 1), mean_losses)
    plt.title('Mean Epoch Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    save_path = output_dir / 'loss_curve.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")

def evaluate_and_visualize(trained_model, args, output_dir: Path):
    """Compares trained and untrained models on a sample image and saves the result."""
    print("\nEvaluating model performance...")
    
    # Use a sample image from the dataset for comparison
    sample_img_path = next(Path(args.image_dir).glob('*.jpg'), None)
    if not sample_img_path:
        print("Could not find a sample image for evaluation. Skipping.")
        return

    # Load original, untrained model
    untrained_sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    untrained_sam.to(args.device)

    predictor_trained = SamPredictor(trained_model)
    predictor_untrained = SamPredictor(untrained_sam)

    image = cv.imread(str(sample_img_path))
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Set images for predictors
    predictor_trained.set_image(image_rgb)
    predictor_untrained.set_image(image_rgb)

    # Create the same point grid prompt used in training
    height, width, _ = image.shape
    num_points, padding = 19, 20
    input_points = np.array(make_point_grid(height, width, num_points, padding))
    input_labels = np.ones(input_points.shape[0], dtype=np.int32)

    # Get predictions
    masks_trained, _, _ = predictor_trained.predict(
        point_coords=input_points, point_labels=input_labels, multimask_output=False
    )
    masks_untrained, _, _ = predictor_untrained.predict(
        point_coords=input_points, point_labels=input_labels, multimask_output=False
    )

    # Visualize and save
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_rgb)
    axes[0].imshow(masks_trained[0], alpha=0.6, cmap='viridis')
    axes[0].set_title('Trained Model')
    axes[0].axis('off')

    axes[1].imshow(image_rgb)
    axes[1].imshow(masks_untrained[0], alpha=0.6, cmap='viridis')
    axes[1].set_title('Untrained Model')
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path = output_dir / 'trained_vs_untrained_comparison.png'
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Comparison image saved to {save_path}")

def main(args):
    """Main function to orchestrate the training process."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Initialize Dataset ---
    # Assumes convert.py has already been run to generate .npy masks
    print(f"Loading data from {args.image_dir} and {args.mask_dir}...")
    dataset = RatDataset(imgs_dir=Path(args.image_dir), masks_dir=Path(args.mask_dir))
    print(f"Found {len(dataset)} images and corresponding masks.")

    # --- 2. Setup SAM Model ---
    print(f"Setting up SAM model ({args.model_type}) from {args.checkpoint}...")
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam_model.to(args.device)

    # --- 3. Preprocess Images ---
    image_keys = [p.name for p in dataset.img_paths]
    transformed_data = preprocess_images_for_sam(image_keys, Path(args.image_dir), sam_model, args.device)

    # --- 4. Setup Optimizer and Loss ---
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # --- 5. Run Training ---
    sam_model.train()
    losses = run_training_loop(sam_model, optimizer, loss_fn, dataset, transformed_data, args.epochs, args.device)

    # --- 6. Save Artifacts ---
    print("\nTraining complete. Saving artifacts...")
    
    # Save model state
    model_save_path = output_dir / f'sam_{args.model_type}_finetuned.pth'
    torch.save(sam_model.state_dict(), model_save_path)
    print(f"Trained model saved to {model_save_path}")

    # Save loss plot
    plot_loss(losses, output_dir)

    # --- 7. Evaluate and Visualize ---
    evaluate_and_visualize(sam_model, args, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune a SAM model on a custom dataset.")
    
    parser.add_argument('--image-dir', type=str, required=True, help='Path to the directory containing images.')
    parser.add_argument('--mask-dir', type=str, required=True, help='Path to the directory containing .npy mask files.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save trained model and artifacts.')
    
    parser.add_argument('--checkpoint', type=str, default='sam_vit_b_01ec64.pth', help='Path to the SAM model checkpoint.')
    parser.add_argument('--model-type', type=str, default='vit_b', help='The type of SAM model to use (e.g., vit_b, vit_l, vit_h).')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='The device to run training on (e.g., "cuda:0" or "cpu").')
    
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay for the optimizer.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs.')

    args = parser.parse_args()
    main(args)
