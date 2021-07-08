import os
import json
from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset


class brain_CT_scan(Dataset):
    """Brain CT Scans dataset."""
    def __init__(self, json_file, root_dir, transform=None, num_classes=15):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(json_file) as f_obj:
            self.dataset_annotations = json.load(f_obj)["questions"]
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset_annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx]['iid']))
        image = np.array(Image.open(img_name)).astype(np.float32)
        image = torch.from_numpy(image).unsqueeze(dim=0)

        classes = self.dataset_annotations[idx]['labels']
        labels = np.zeros(self.num_classes).astype(np.uint8)
        labels[classes] = 1

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'labels': torch.tensor(labels, dtype=torch.float32)
        }
