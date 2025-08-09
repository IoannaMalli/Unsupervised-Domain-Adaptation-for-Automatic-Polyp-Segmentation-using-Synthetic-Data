import os
import torch

class CheckpointHook:
    def __init__(self, model, optimizer, scheduler, save_dir, save_every=100):
        """
        A hook to save model checkpoints every 'save_every' iterations.

        Args:
            model (torch.nn.Module): The model being trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training.
            save_dir (str): Directory where checkpoints will be saved.
            save_every (int): Save a checkpoint every 'save_every' iterations.
        """
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.save_every = save_every
        self.scheduler = scheduler

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

    def maybe_save_checkpoint(self, iteration):
        """Save a checkpoint with the current iteration in the filename."""
        checkpoint_filename = f"checkpoint_iter_{iteration:4d}.pth"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_filename)

        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }, checkpoint_path)

        print(f"Checkpoint saved at iteration {iteration}: {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path)
    model.to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    iteration = checkpoint['iteration']
    print(f"Checkpoint loaded from iteration {iteration}")
    return iteration