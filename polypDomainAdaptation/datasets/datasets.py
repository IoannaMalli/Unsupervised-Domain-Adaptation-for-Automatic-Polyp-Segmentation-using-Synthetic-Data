import Image
import os 
import numpy as np
from torch.utils.data import Dataset


class kvasirsegDataset(Dataset):
    def __init__(self,img_list,mask_list,img_path, mask_path, transform=None):
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform
        self.img_path = img_path
        self.mask_path = mask_path

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
      #get image path
        img_path = os.path.join(self.img_path,self.img_list[idx])
        mask_path = os.path.join(self.mask_path,self.mask_list[idx])
      #image
        img = Image.open(img_path)
        mask = Image.open(mask_path).convert("L")
        # print(f'unique values in mask {np.unique(np.array(mask))}')
      # turn to array
        img = np.array(img)
        mask = np.array(mask)
        mask = (mask > 127)*255
        # print(f'unique values in mask (threshold) {np.unique(mask)}')
        mask = mask / 255.0
        # print(f'unique values in mask (normalized) {np.unique(mask)}')
        if self.transform is not None:
            augmentations = self.transform(image=img,mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]
        mask = mask.unsqueeze(0)


        return {
            "img": img,
            "mask": mask
        }
    

import tifffile

class CVC_ClinicDB(Dataset):
    def __init__(self,img_list,mask_list,img_path, mask_path, transform=None):
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform
        self.img_path = img_path
        self.mask_path = mask_path

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self,idx):
      #get image path
        img_path = os.path.join(self.img_path,self.img_list[idx])
        mask_path = os.path.join(self.mask_path,self.mask_list[idx])
      #image
        img = tifffile.imread(img_path)
        mask = Image.open(mask_path).convert('L') # Grayscale mask
        # print(f'unique values in mask {np.unique(np.array(mask))}')
      # turn to array
        img = np.array(img)
        mask = np.array(mask)
        mask = (mask > 127)*255
        # print(f'unique values in mask (threshold) {np.unique(mask)}')
        mask = mask / 255.0
        # print(f'unique values in mask (normalized) {np.unique(mask)}')
        if self.transform is not None:
            augmentations = self.transform(image=img,mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]
        mask = mask.unsqueeze(0)

        return {
            "img": img,
            "mask": mask
        }


    
class SynthColonDataset(Dataset):
    def __init__(self, img_list, mask_list, img_path, mask_path, transform=None):
        self.img_list = img_list
        self.mask_list = mask_list
        self.transform = transform
        self.img_path = img_path
        self.mask_path = mask_path

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.img_list[idx])
        mask_path = os.path.join(self.mask_path, self.mask_list[idx])

        try:
            # Try loading the image and mask
            img = Image.open(img_path).convert("RGB")  # Ensure it's in RGB mode
            mask = Image.open(mask_path).convert('L') # Grayscale mask
            # Convert to array
            img = np.array(img)
            mask = np.array(mask)
            mask = (mask > 127)*255
            mask = np.array(mask) / 255.0  # Normalize mask


            # Apply transformations (if any)
            if self.transform is not None:
                augmentations = self.transform(image=img, mask=mask)
                img = augmentations["image"]
                mask = augmentations["mask"]
            # Convert mask to tensor
            mask = mask.unsqueeze(0)

            return {
            "img": img,
            "mask": mask
            }

        except Exception as e:
            print(f" Skipping corrupted image: {img_path}. Error: {e}")
            return None  # Return None for corrupted images