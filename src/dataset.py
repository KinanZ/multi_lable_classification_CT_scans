import os
import json
from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset


class brain_CT_scan(Dataset):
    """Brain CT Scans dataset."""

    def __init__(self, json_file, root_dir, transform=None, num_channels=3, num_classes=15):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            num_channels (int): if 1 -> one slice is the input for the model,
                if 3 -> the previous and post slices are stacked to the main slice and become a 3-channel input for the model.
            num_classes (int): number of categories in the dataset
        """
        with open(json_file) as f_obj:
            self.dataset_annotations = json.load(f_obj)["questions"]
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.num_channels = num_channels

    def __len__(self):
        return len(self.dataset_annotations)

    def __getitem__(self, idx):
        img_iid = int(self.img_list[idx][:-4])
        if self.num_channels == 1:
            img_name = os.path.join(self.root_dir, self.img_list[idx])
            image = np.array(Image.open(img_name)).astype(np.float32)
            image = torch.from_numpy(image).unsqueeze(dim=0)
        elif self.num_channels == 3:
            img_name_mid = os.path.join(self.root_dir, self.img_list[idx])
            try:
                img_iid_pre = int(self.img_list[idx - 1][:-4])  # -'.jpg'
                if img_iid_pre == img_iid - 1:
                    img_name_pre = os.path.join(self.root_dir, self.img_list[idx - 1])
                else:
                    img_name_pre = os.path.join(self.root_dir, self.img_list[idx])
            except:
                # if idx == 0
                img_name_pre = os.path.join(self.root_dir, self.img_list[idx])

            try:
                img_iid_post = int(self.img_list[idx + 1][:-4])  # -'.jpg'
                if img_iid_post == img_iid + 1:
                    img_name_post = os.path.join(self.root_dir, self.img_list[idx + 1])
                else:
                    img_name_post = os.path.join(self.root_dir, self.img_list[idx])
            except:
                # if idx == len(self.img_list)
                img_name_post = os.path.join(self.root_dir, self.img_list[idx])

            image_mid = np.array(Image.open(img_name_mid)).astype(np.float32)
            image_pre = np.array(Image.open(img_name_pre)).astype(np.float32)
            image_post = np.array(Image.open(img_name_post)).astype(np.float32)
            image = np.dstack((image_pre, image_mid, image_post))
            image = torch.from_numpy(image).permute(2, 0, 1)

        classes = self.dataset_annotations[idx]['labels']
        labels = np.zeros(self.num_classes).astype(np.uint8)
        labels[classes] = 1

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'labels': torch.tensor(labels, dtype=torch.float32)
        }
