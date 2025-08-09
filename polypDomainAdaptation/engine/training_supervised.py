import os
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

def train_segmentor(
    model,
    train_dataloader,
    val_dataloader,
    num_epochs,
    criterion,
    optimizer,
    scheduler,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    validate_every_epoch=True,
    work_dir='/content/drive/MyDrive/DAFormer_1/work_dir/custom_thesis/mitb5/source'
):
    """
    Train the segmentor model with validation and loss tracking.

    Args:
        model (nn.Module): The segmentor model to train.
        train_dataloader (DataLoader): Dataloader for training data.
        val_dataloader (DataLoader): Dataloader for validation data.
        num_epochs (int): Number of epochs to train.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device to run the training on ('cuda' or 'cpu').
        validate_every_epoch (bool): Whether to validate after every epoch.

    Returns:
        None (Metrics are saved to a file)
    """
    model = model.to(device)

    # Create a new directory named with the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join(work_dir, current_time)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Path to metrics file
    metrics_file = os.path.join(checkpoint_dir, 'metrics.txt')

    # Write the header of the file
    with open(metrics_file, 'w') as f:
        f.write("Epoch,Train Loss,Val Loss,mIoU\n")

    # Create the checkpoint hook
    # checkpoint_hook = CheckpointHook(model, optimizer, checkpoint_dir)

    # Example scheduler (optional if you have your own)
    # Make sure to replace WarmupPolyLR with whatever LR scheduler you prefer
    # If you don't need it, you can remove it
    #scheduler = WarmupPolyLR(optimizer, max_iter=50 * 176, power=1.0, warmup_iters=500, warmup_factor=0.1)

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]")

        for batch in pbar:
            inputs, targets = batch  # Adjust keys to match your dataset
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            train_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # Update progress bar
            pbar.set_postfix({'Loss': loss.item()})

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation Phase
        avg_val_loss, miou = 0.0, 0.0
        if validate_every_epoch and val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            total_iou = 0.0
            total_dice =0.0

            with torch.no_grad():
                pbar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]")
                for batch in pbar:
                    inputs, targets = batch  # Adjust keys to match your dataset
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Forward pass
                    outputs = model(inputs)

                    # Compute loss
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    # Compute IoU
                    output_probs = torch.sigmoid(outputs).squeeze(1)  # shape (N, H, W)
                    predicted_masks = (output_probs > 0.5).float()    # shape (N, H, W)
                    iou = calculate_miou(predicted_masks.cpu().numpy(), targets.cpu().numpy())
                    dice = calculate_dice(predicted_masks.cpu().numpy(), targets.cpu().numpy())
                    total_iou += iou
                    total_dice += dice

                    # Update progress bar
                    pbar.set_postfix({'Loss': loss.item(), 'IoU': iou})

            avg_val_loss = val_loss / len(val_dataloader)
            miou = total_iou / len(val_dataloader)
            mdice = total_dice / len(val_dataloader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, mIoU: {miou:.4f}, mDice: {mdice:.4f}")

        # Save the checkpoint
        # checkpoint_hook.save_checkpoint(epoch + 1

        # Save metrics to file
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch + 1},{avg_train_loss:.4f},{avg_val_loss:.4f},{miou:.4f},{mdice:.4f}\n")


        # checkpoint_hook.save_checkpoint(epoch + 1)

    # Optionally save a final checkpoint as well
    final_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs
    }, final_checkpoint_path)
    print(f"Final checkpoint saved at {final_checkpoint_path}")
    print(f"Metrics saved at {metrics_file}")
