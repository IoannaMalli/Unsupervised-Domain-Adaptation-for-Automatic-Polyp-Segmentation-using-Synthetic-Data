import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(p=0.5, brightness=(0.6, 1.6), contrast=0.2, saturation=0.1, hue=0.01),
    A.Affine(p=0.5, scale=(0.5, 1.5), translate_percent=0.125, rotate=90, interpolation=cv2.INTER_NEAREST),
    A.ElasticTransform(p=0.5, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

val_transform = A.Compose([
    A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
    ToTensorV2()
], additional_targets={'mask': 'mask'})
