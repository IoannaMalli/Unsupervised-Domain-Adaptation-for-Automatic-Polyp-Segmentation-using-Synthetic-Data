from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score

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