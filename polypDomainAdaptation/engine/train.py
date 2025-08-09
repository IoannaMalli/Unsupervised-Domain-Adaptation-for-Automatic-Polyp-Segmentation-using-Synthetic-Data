from collections import defaultdict
import tqdm
from datetime import datetime
import torch
import os

from Hooks import ValidationHook, CheckpointHook
from utils import init_ema_weights
from .step import train_step

def train_segmentor(
    model,
    train_dataloader,
    val_dataloader=None,  # Validation DataLoader (optional)
    num_epochs=2,
    criterion=None,
    optimizer=None,
    scheduler=None,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    validate_every=100,   # Flag to control validation frequency
    work_dir='/content/drive/MyDrive/DAFormer_1/work_dir/full_dataset/mitb5/without_GAP'
    ):
    """
    Train the segmentor model with optional validation and loss tracking.

    Args:
        model (nn.Module): The segmentor model to train.
        train_dataloader (DataLoader): Dataloader for training data.
        val_dataloader (DataLoader, optional): Dataloader for validation data.
        num_epochs (int): Number of epochs to train.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device to run the training on ('cuda' or 'cpu').
        validate_every_epoch (bool): Whether to validate after every epoch.
        work_dir (str): Directory to save checkpoints and logs.

    Returns:
        dict: Dictionary containing training and validation loss history.
    """
    model = model.to(device)


    # Create a directory for saving checkpoints
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join(work_dir, current_time)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Checkpoint hook
    iteration = 0
    checkpoint_hook = CheckpointHook(model, optimizer, scheduler, checkpoint_dir, save_every=100)

    # Initialize EMA model
    ema_model = init_ema_weights(model)
    ema_model = ema_model.to(device)

    # Initialize validation hook
    val_hook = ValidationHook(model, ema_model, val_dataloader, criterion,  checkpoint_dir, device, validate_every)

    # Loss history
    train_total_loss_history = []
    train_source_loss_history = []
    train_target_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        epoch_total_loss = 0.0
        epoch_source_loss = 0.0
        epoch_target_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]")

        for batch in pbar:
            train_total_loss, train_source_loss, train_target_loss = train_step(
                batch,
                iteration,
                device,
                model,
                ema_model,
                criterion,
                optimizer,
                scheduler
            )

            # Accumulate losses
            epoch_total_loss += train_total_loss
            epoch_source_loss += train_source_loss
            epoch_target_loss += train_target_loss

            if (iteration + 1) % checkpoint_hook.save_every == 0:
              checkpoint_hook.maybe_save_checkpoint( iteration)

            # Update progress bar
            pbar.set_postfix({
            'Total Loss': train_total_loss,
            'Source Loss': train_source_loss,
            'Target Loss': train_target_loss
            })


            #validation hook
            if val_dataloader:
              val_hook.maybe_validate()
              val_hook.update()

            iteration += 1

        # Compute average losses for the epoch
        avg_train_loss = epoch_total_loss / len(train_dataloader)
        avg_source_loss = epoch_source_loss / len(train_dataloader)
        avg_target_loss = epoch_target_loss / len(train_dataloader)

        train_total_loss_history.append(avg_train_loss)
        train_source_loss_history.append(avg_source_loss)
        train_target_loss_history.append(avg_target_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, "
              f"Source Loss: {avg_source_loss:.4f}, Target Loss: {avg_target_loss:.4f}")

        # # Save checkpoint
        # checkpoint = {
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        #     'epoch': epoch + 1,
        #     'train_loss_history': train_total_loss_history,
        #     'val_loss_history': val_loss_history
        # }
        # checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        # torch.save(checkpoint, checkpoint_path)

    print(f"Training completed. Checkpoints saved in {checkpoint_dir}")

    return {
        'train_loss': train_total_loss_history,
        'val_loss': val_loss_history
    }