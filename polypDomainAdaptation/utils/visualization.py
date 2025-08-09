from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

def calculate_miou(predicted, ground_truth, num_classes=1):
    """
    Calculate mean Intersection over Union (mIoU).

    Args:
        predicted (np.ndarray): Predicted mask.
        ground_truth (np.ndarray): Ground truth mask.
        num_classes (int): Number of classes.

    Returns:
        float: mIoU value.
    """
    predicted_flat = predicted.flatten()
    ground_truth_flat = ground_truth.flatten()


    miou = jaccard_score(ground_truth_flat, predicted_flat, average='binary', labels=np.arange(num_classes))
    return miou

def calculate_dice(predicted, ground_truth, num_classes=1):
    """
    Calculate Dice Score (F1 Score).

    Args:
        predicted (np.ndarray): Predicted mask.
        ground_truth (np.ndarray): Ground truth mask.
        num_classes (int): Number of classes.

    Returns:
        float: Dice Score value.
    """
    predicted_flat = predicted.flatten()
    ground_truth_flat = ground_truth.flatten()

    dice_score = f1_score(ground_truth_flat, predicted_flat, average='binary', labels=np.arange(num_classes))
    return dice_score

def evaluate_all_images(model, image_dir, mask_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate the segmentor on all test images and calculate the average mIoU.

    Args:
        model (nn.Module): Trained segmentor model.
        image_dir (str): Directory containing test images.
        mask_dir (str): Directory containing ground truth masks.
        device (str): Device to run the model on.

    Returns:
        float: Average mIoU across all images.
    """
    # Set model to evaluation mode
    model.eval()
    device = torch.device(device)
    model = model.to(device)

    # List all image and mask files
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    # Check if the number of images and masks matches
    assert len(image_files) == len(mask_files), "Number of images and masks must match."

    total_miou = 0.0
    num_files = len(image_files)

    for i, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
        # Load and preprocess the image
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((512, 512), resample=Image.NEAREST)

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        image_tensor = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            # Perform inference
          output = model(image_tensor)
          output_probs = torch.sigmoid(output)

          predicted_mask = (output_probs > 0.5).squeeze(1).cpu().numpy()
          total_elements = predicted_mask.size  # Total number of elements in the array
          num_ones = np.sum(predicted_mask == 1)  # Count the elements equal to 1
          percentage_ones = (num_ones / total_elements) * 100  # Calculate percentage


          # Convert ground truth mask to numpy array
          ground_truth_mask = np.array(mask)
          ground_truth_mask = (ground_truth_mask > 127)*255
          ground_truth_mask = np.array(ground_truth_mask) / 255.0

          # Calculate mIoU
          miou = calculate_miou(predicted_mask, ground_truth_mask)


          total_miou += miou



    # Compute the average mIoU
    average_miou = total_miou / num_files
    print(f"Average mIoU across all images: {average_miou:.4f}")

    return average_miou