import os
import json
from PIL import Image
import numpy as np
import torch
import random

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from my_utils import crop_show_augment

from my_utils import crop_show_augment


class brain_CT_scan(Dataset):
    """Brain CT Scans dataset."""

    def __init__(self, json_file, root_dir, transform=None, stack_pre_post=True, num_classes=15, bbox_aug=False):
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
        self.bbox_aug = bbox_aug

        self.y = np.zeros((len(self.dataset_annotations), self.num_classes)).astype(np.uint8)
        for i in range(len(self.dataset_annotations)):
            classes = self.dataset_annotations[i]['labels']
            self.y[i][classes] = 1


    def __len__(self):
        return len(self.dataset_annotations)

    def __getitem__(self, idx):
        if not self.stack_pre_post:
            img_name = os.path.join(self.root_dir, '{0:07d}.jpg'.format(self.dataset_annotations[idx]['iid']))
            image = np.array(Image.open(img_name)).astype(np.float32)
            image = np.dstack((image, image, image))
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

        classes = self.dataset_annotations[idx]['labels']
        labels = np.zeros(self.num_classes).astype(np.uint8)
        labels[classes] = 1
        bboxes = self.dataset_annotations[idx]['bboxes']

        if self.bbox_aug:
            image = crop_show_augment(image, labels, bboxes)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'labels': torch.tensor(labels, dtype=torch.float32)
        }


class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, labels, indices=None, class_choice="least_sampled"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
            class_choice: a string indicating how class will be selected for every
            sample:
                "least_sampled": class with the least number of sampled labels so far
                "random": class is chosen uniformly at random
                "cycle": the sampler cycles through the classes sequentially
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))

        self.num_classes = self.labels.shape[1]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        return class_

    def __len__(self):
        return len(self.indices)