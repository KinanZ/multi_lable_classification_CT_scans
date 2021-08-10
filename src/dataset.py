import os
import json
from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset


class brain_CT_scan(Dataset):
    """Brain CT Scans dataset."""

    def __init__(self, json_file, root_dir, transform=None, stack_pre_post=True, num_classes=15):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            stack_pre_post (bool): if True -> the previous and post slices are stacked to the main slice and become a 3-channel input for the model.
            num_classes (int): number of categories in the dataset
        """
        with open(json_file) as f_obj:
            self.dataset_annotations = json.load(f_obj)["questions"]
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.stack_pre_post = stack_pre_post

    def __len__(self):
        return len(self.dataset_annotations)

    def __getitem__(self, idx):
        if not self.stack_pre_post:
            img_name = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx]['iid']))
            image = np.array(Image.open(img_name)).astype(np.float32)
            image = np.dstack((image, image, image))
            image = torch.from_numpy(image).permute(2, 0, 1)
        else:
            img_name_mid = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx]['iid']))
            try:
                img_name_pre = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx-1]['iid']))
            except:
                # if idx == 0
                img_name_pre = img_name_mid

            try:
                img_name_post = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx+1]['iid']))
            except:
                # if idx == len(self.img_list)
                img_name_post = img_name_mid

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
