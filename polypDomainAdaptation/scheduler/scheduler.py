import torch

class WarmupDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters=100, decay_iters=2000,
                 min_lr=1e-6, peak_lr=3e-5, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.decay_iters = decay_iters
        self.min_lr = min_lr
        self.peak_lr = peak_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]  # Not used, but stored

        super(WarmupDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Linear Warmup: From min_lr to peak_lr
            warmup_factor = self.last_epoch / self.warmup_iters
            current_lr = self.min_lr + warmup_factor * (self.peak_lr - self.min_lr)
        elif self.last_epoch < self.decay_iters:
            # Polynomial Decay: From peak_lr to min_lr
            decay_progress = (self.last_epoch - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
            decay_factor = (1 - decay_progress) ** 1.0  # power=1.0 = linear decay
            current_lr = self.min_lr + decay_factor * (self.peak_lr - self.min_lr)
        else:
            # Constant min_lr after decay_iters
            current_lr = self.min_lr

        return [current_lr for _ in self.optimizer.param_groups]

from torch.optim import Optimizer

class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, max_iter: int, power: float = 1.0, warmup_iters: int = 500, warmup_factor: float = 0.1, last_epoch: int = -1):
        self.max_iter = max_iter
        self.power = power
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Linear warmup
            alpha = self.last_epoch / float(self.warmup_iters)
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            progress = (self.last_epoch - self.warmup_iters) / (self.max_iter - self.warmup_iters)
            decay_factor = (1 - progress) ** self.power
            return [base_lr * decay_factor for base_lr in self.base_lrs]

class ConstantThenDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, constant_iters=100, decay_iters=1900,
                 min_lr=6e-7, max_lr=1e-5, last_epoch=-1):
        self.constant_iters = constant_iters
        self.decay_iters = decay_iters
        self.min_lr = min_lr
        self.max_lr = max_lr

        super(ConstantThenDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.constant_iters:
            # Constant learning rate
            current_lr = self.max_lr
        elif self.last_epoch < self.constant_iters + self.decay_iters:
            # Polynomial decay
            decay_progress = (self.last_epoch - self.constant_iters) / self.decay_iters
            decay_factor = (1 - decay_progress) ** 1.0  # Linear decay
            current_lr = self.min_lr + decay_factor * (self.max_lr - self.min_lr)
        else:
            # Hold min_lr constant
            current_lr = self.min_lr

        return [current_lr for _ in self.optimizer.param_groups]
