from utils.helper_functions import collate_fn, load_image_mask_lists
from utils.paths import data_paths

from datasets.datasets import SynthColonDataset, kvasirsegDataset
from datasets.transforms import train_transform, val_transform
from datasets.uda_dataset import UDADataset
from model.encoder import mit_b5
from model.decoder import DAFormerHead
from model.segmentor import Segmentor
from engine.training_supervised import train_segmentor

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# Load datasets
src_train_imgs, src_val_imgs, src_train_masks, src_val_masks = load_image_mask_lists(
    data_paths["synthcolon"]["images"], data_paths["synthcolon"]["masks"], split=0.2
)

train_dataset = SynthColonDataset(src_train_imgs, src_train_masks, data_paths["synthcolon"]["images"], data_paths["synthcolon"]["masks"], train_transform)

val_dataset = SynthColonDataset(src_val_imgs, src_val_masks, data_paths["synthcolon"]["images"], data_paths["synthcolon"]["masks"], val_transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2)

# Model
encoder = mit_b5()
decoder = DAFormerHead(in_channels=[64, 128, 320, 512], embed_dims=[64, 128, 256, 256], dilations=[1, 6, 12, 18], num_classes=1)
segmentor = Segmentor(encoder, decoder)

criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
optimizer = AdamW(segmentor.parameters(), lr=1e-5, weight_decay=1e-5)

train_segmentor(segmentor, train_loader, val_loader, 6, criterion, optimizer, None, device='cuda' if torch.cuda.is_available() else 'cpu')
