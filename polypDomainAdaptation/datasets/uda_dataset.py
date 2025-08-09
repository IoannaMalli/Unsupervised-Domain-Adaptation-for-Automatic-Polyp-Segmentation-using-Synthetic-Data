from torch.utils.data import Dataset

class UDADataset(Dataset):
    def __init__(self, source_dataset, target_dataset):
        """
        Args:
            source_dataset (Dataset): Labeled source dataset.
            target_dataset (Dataset): Unlabeled target dataset.
        """
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

    def __len__(self):
        # Define the length based on the smaller of the two datasets
        return len(self.source_dataset)

    def __getitem__(self, idx):
        # Fetch source data
        source_data = self.source_dataset[idx]

        # Cycle through target dataset
        target_idx = idx % len(self.target_dataset)
        target_data = self.target_dataset[target_idx]

        return {
            "source_img": source_data["img"],
            "source_label": source_data["mask"],
            "target_img": target_data["img"],
            "target_gt": target_data.get("mask", None),
        }

