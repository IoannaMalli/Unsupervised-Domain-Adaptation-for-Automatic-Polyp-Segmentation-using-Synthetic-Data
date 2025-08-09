import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os 
import torch

def get_parameter_norm(model):
    total_norm = 0.0
    for param in model.parameters():
        # Using .data to access the underlying tensor.
        param_norm = param.data.norm(2)  # L2 norm of the parameter tensor
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5  # square root of the sum of squares
    return total_norm

def get_gradient_norm(model):
    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)  # L2 norm of the gradient tensor
            total_grad_norm += grad_norm.item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    return total_grad_norm


def visualize_classmix(source_image, source_label, target_image, pseudo_label, mixed_image, mixed_label):
    """
    Function to visualize the images and labels before and after ClassMix.
    """

    def tensor_to_numpy(tensor):
        """Convert PyTorch tensor to NumPy format for visualization"""
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

    def mask_to_numpy(mask):
        """Convert label mask to NumPy format for visualization"""
        return mask.detach().cpu().numpy()  # No need for channel dimension

    # Convert tensors to NumPy format
    source_img_np = tensor_to_numpy(source_image)
    target_img_np = tensor_to_numpy(target_image)
    mixed_img_np = tensor_to_numpy(mixed_image)

    source_label_np = mask_to_numpy(source_label)
    pseudo_label_np = mask_to_numpy(pseudo_label)
    mixed_label_np = mask_to_numpy(mixed_label)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original Images
    axes[0, 0].imshow(source_img_np)
    axes[0, 0].set_title("Source Image")

    axes[0, 1].imshow(target_img_np)
    axes[0, 1].set_title("Target Image")

    axes[0, 2].imshow(mixed_img_np)
    axes[0, 2].set_title("Mixed Image (ClassMix)")

    # Row 2: Corresponding Labels
    axes[1, 0].imshow(source_label_np, cmap='gray')
    axes[1, 0].set_title("Source Label (Ground Truth)")

    axes[1, 1].imshow(pseudo_label_np, cmap='gray')
    axes[1, 1].set_title("Pseudo Label (Target Image)")

    axes[1, 2].imshow(mixed_label_np, cmap='gray')
    axes[1, 2].set_title("Mixed Label (ClassMix)")

    plt.tight_layout()
    plt.show()

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return None if not batch else torch.utils.data.dataloader.default_collate(batch)

def list_sorted_files(path):
    return sorted(os.listdir(path))

def load_image_mask_lists(img_path, mask_path, split=None, random_state=42):
    imgs = list_sorted_files(img_path)
    masks = list_sorted_files(mask_path)
    if split:
        return train_test_split(imgs, masks, test_size=split, random_state=random_state)
    return imgs, masks