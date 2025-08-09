import os
import torch
from collections import defaultdict

from engine import val_step

class ValidationHook:
    def __init__(self, model, ema_model, val_dataloader, criterion, checkpoint_dir, device='cuda', validate_every=50):
        """
        Custom hook to validate every N iterations.

        Args:
            model (nn.Module): Model to evaluate.
            ema_model (nn.Module): Exponential Moving Average model (optional).
            val_dataloader (DataLoader): Validation dataloader.
            criterion (nn.Module): Loss function.
            checkpoint_dir (str): Directory for saving checkpoints/metrics.
            device (str): Device for validation ('cuda' or 'cpu').
            validate_every (int): Run validation every N iterations.
        """
        self.model = model
        self.ema_model = ema_model
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.device = device
        self.validate_every = validate_every
        self.iteration = 0  # Track the current iteration
        self.val_loss_history = []
        self.checkpoint_dir = checkpoint_dir

        # File path for evaluation metrics
        self.metrics_path = os.path.join(self.checkpoint_dir, "evaluation_metrics.txt")

        # Write header if file is newly created
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w") as f:
                f.write("Iteration, Target Loss, mIoU, Dice\n")  # CSV format header

    def log_evaluation_results(self, target_loss, miou, dice):
        """Logs validation metrics to a text file."""
        with open(self.metrics_path, "a") as f:
            f.write(f"{self.iteration}, {target_loss:.4f}, {miou:.4f}, {dice:.4f}\n")

    def maybe_validate(self):
        """Runs validation if the iteration count reaches the threshold."""
        if self.iteration % self.validate_every == 0 and self.iteration != 0 :
            print(f"\n[Validation] Running at iteration {self.iteration}...")
            self.model.eval()

            # Dictionary to accumulate all metric sums
            val_metrics = defaultdict(float)

            with torch.no_grad():
                for val_batch in self.val_dataloader:
                    metrics = val_step(val_batch, self.device, self.model, self.ema_model, self.criterion)

                    # Accumulate each metric
                    for k, v in metrics.items():
                        val_metrics[k] += v

            # Now average each metric across all validation batches
            num_batches = len(self.val_dataloader)
            for k in val_metrics.keys():
                val_metrics[k] /= num_batches

            # Extract specific metrics for logging
            target_loss = val_metrics.get('target_loss', 0.0)
            miou = val_metrics.get('iou', 0.0)   # Assuming 'iou' key exists
            dice = val_metrics.get('dice', 0.0)  # Assuming 'dice' key exists

            # Log validation results
            self.log_evaluation_results(target_loss, miou, dice)

            # Convert metrics to a single string for printing
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            self.val_loss_history.append(val_metrics.get('source_loss', 0.0))  # Track source_loss if applicable

            print(f"\n[Validation] Iteration {self.iteration} | {metrics_str}")

    def update(self):
        """Increment iteration counter."""
        self.iteration += 1
