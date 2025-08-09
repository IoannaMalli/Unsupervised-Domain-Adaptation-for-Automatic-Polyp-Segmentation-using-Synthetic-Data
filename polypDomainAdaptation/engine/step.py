import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

from utils import (
    calculate_miou,
    calculate_dice,
    adjust_alpha,
    adjust_lambda_target,
    adjust_pseudo_threshold,
    update_ema_weights
)


def apply_strong_transforms(image, mask, flip_prob=0.5, jitter_prob=0.5, blur_prob=0.5):
    """
    Apply strong transforms to both the given image and mask.
    - Random horizontal flip (applied to both image and mask)
    - Color jitter (applied only to the image)
    - Gaussian blur (applied only to the image)

    Args:
        image (torch.Tensor): The input image (C, H, W).
        mask (torch.Tensor): The corresponding mask (H, W) or (1, H, W).
        flip_prob (float): Probability of horizontal flipping.
        jitter_prob (float): Probability of applying color jitter.
        blur_prob (float): Probability of applying Gaussian blur.

    Returns:
        torch.Tensor: Transformed image.
        torch.Tensor: Transformed mask.
    """

    # Random horizontal flip
    if random.random() < flip_prob:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random color jitter (applied only to the image)
    if random.random() < jitter_prob:
        jitter_transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        image = jitter_transform(image)

    # Random Gaussian blur (applied only to the image)
    if random.random() < blur_prob:
        blur_transform = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
        image = blur_transform(image)

    return image, mask

def train_step(batch,
               iteration,
               device,
               source_segmentor,
               ema_model,
               criterion,
               optimizer,
               scheduler):

    with torch.autograd.set_detect_anomaly(True):

        # Step 1: Train on Source Images (Supervised)
        inputs, gt_semantic_seg, target_imgs = batch['source_img'], batch['source_label'], batch['target_img']
        inputs, gt_semantic_seg, target_imgs = inputs.to(device), gt_semantic_seg.to(device), target_imgs.to(device)

        # Forward pass on source
        source_segmentor.train()
        outputs = source_segmentor(inputs)
        source_loss = criterion(outputs, gt_semantic_seg)  # Supervised loss
        source_loss = source_loss.mean()

        # Adjust alpha and lambda for domain adaptation
        alpha = adjust_alpha(iteration)
        lambda_target = adjust_lambda_target(iteration)

        # Adjust pseudo-label threshold dynamically
        pseudo_threshold = adjust_pseudo_threshold(iteration)

        # Step 2: Generate Pseudo-Labels for Target Images
        with torch.no_grad():
            # Update EMA model
            update_ema_weights(source_segmentor, ema_model, alpha)

            # Pseudo labels from EMA model
            ema_model.eval()
            pseudo_logits = ema_model(target_imgs).detach()
            pseudo_probs = torch.sigmoid(pseudo_logits).squeeze(1)  # [N, H, W]
            pseudo_labels_binary = (pseudo_probs.ge(0.5)).long()  # Binary pseudo-labels

            # Confidence mask for pseudo-labels
            # print('pseudo labels sum', torch.sum(pseudo_labels_binary).item())
            # print('pseudo labels conf', torch.sum(pseudo_probs.ge(pseudo_threshold)).item())
            unlabeled_weight = torch.sum(pseudo_probs.ge(pseudo_threshold)).item() / torch.sum(pseudo_labels_binary + 1e-6).item()
            pixelWiseWeight = unlabeled_weight * torch.ones(pseudo_probs.shape).cuda() # [N, H, W]

        # Step 3: ClassMix Augmentation (Bidirectional)
        if torch.rand(1).item() > 0.5:

            # Standard ClassMix: Paste source polyps onto the target background
            MixMask = (gt_semantic_seg == 1).float()  # Select polyp pixels
            mixed_image = inputs * MixMask + target_imgs * (1 - MixMask)
            mixed_label = gt_semantic_seg  + pseudo_labels_binary.unsqueeze(1) * (1 - MixMask)
            mixed_label = mixed_label.squeeze(1)
            #visualize_classmix(inputs[0], gt_semantic_seg[0].squeeze(0), target_imgs[0], pseudo_labels_binary[0], mixed_image[0], mixed_label[0])

        else:
            # Reverse ClassMix: Paste target pseudo-polyps onto the source background
            MixMask = (pseudo_labels_binary == 1).float()  # Select target polyp pixels
            MixMask_rgb = MixMask.unsqueeze(1).repeat(1, 3, 1, 1)  # Shape: [2, 3, 512, 512]
            mixed_image = target_imgs * MixMask_rgb + inputs * (1 - MixMask_rgb)
            mixed_label = pseudo_labels_binary * MixMask + gt_semantic_seg.squeeze(1) * (1 - MixMask)
            #visualize_classmix(inputs[0], gt_semantic_seg[0].squeeze(0), target_imgs[0], pseudo_labels_binary[0], mixed_image[0], mixed_label[0])

        # Mixed Image Augmentation
        mixed_image, mixed_label = apply_strong_transforms(mixed_image,mixed_label, flip_prob=0.5, jitter_prob=0.8, blur_prob=0.5)

        # Step 4: Train on Mixed Images (Unsupervised Loss)
        mixed_outputs = source_segmentor(mixed_image)  # Forward pass on mixed images
        target_loss = criterion(
            mixed_outputs,  # Predicted probabilities
            mixed_label.unsqueeze(1)  # Mixed pseudo-labels
        )



        # Apply confidence-based weighting
        target_loss_weighted = torch.mean(target_loss * pixelWiseWeight)

        # Scale by lambda and backprop
        target_loss_weighted = lambda_target * target_loss_weighted
        total_loss = source_loss + target_loss_weighted

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Adjust pseudo-label threshold dynamically
        pseudo_threshold = adjust_pseudo_threshold(iteration)

        # Log metrics
        with torch.no_grad():
            source_loss = source_loss.item()
            target_loss = target_loss.mean().item()
            total_loss = total_loss.item()

    return total_loss, source_loss, target_loss

def val_step(
    batch,
    device,
    model,
    ema_model,
    criterion
    ):

    model.eval()
    if ema_model is not None:
        ema_model.eval()

    metrics = {}

    with torch.no_grad():
        # Step 1: Target Loss
        target_imgs, target_masks = batch['img'], batch['mask']
        target_imgs, target_masks = target_imgs.to(device), target_masks.to(device)

        outputs = model(target_imgs)
        # print(f'\n parameter norm (segmentor val) {get_parameter_norm(model)}')
        # print(f'gradient norm (segmentor val) {get_gradient_norm(model)}')
        target_loss = criterion(outputs, target_masks)  # Unsupervised loss
        target_loss = torch.mean(target_loss)
        metrics['target_loss'] = target_loss.item()

        # Step 2: Intersection over Union
        output_probs = torch.sigmoid(outputs).squeeze(1)  # shape (N, H, W)
        predicted_masks = (output_probs > 0.5).float()    # shape (N, H, W)
        iou = calculate_miou(predicted_masks.cpu().numpy(), target_masks.cpu().numpy())
        metrics['iou'] = iou

        # Step 3: Dice Score
        dice = calculate_dice(predicted_masks.cpu().numpy(), target_masks.cpu().numpy())
        metrics['dice'] = dice


    return metrics

