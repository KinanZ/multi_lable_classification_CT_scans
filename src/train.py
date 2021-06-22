import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from engine import train, validate
from dataset import brain_CT_scan, train_transforms, valid_transforms
from my_model import resnet_model


json_file_path = "/misc/lmbraid19/argusm/CLUSTER/multimed/NSEG2015_2/train.json"
images_path = "/misc/lmbraid19/argusm/CLUSTER/multimed/NSEG2015_2/JPEGImages/"

train_dataset = brain_CT_scan(json_file_path, images_path, train_transforms)
valid_dataset = brain_CT_scan(json_file_path, images_path, valid_transforms)

batch_size = 4
validation_split = .15
shuffle_dataset = True
random_seed= 101

# Creating data indices for training and validation splits:
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#intialize the model
model = resnet_model(num_classes=15, pretrained=True, requires_grad=True).to(device)
# learning parameters
lr = 0.0001
epochs = 2
batch_size = 6
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

# start the training and validation
train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_loader, optimizer, criterion, train_dataset, device
    )
    valid_epoch_loss = validate(
        model, valid_loader, criterion, valid_dataset, device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {valid_epoch_loss:.4f}')
    
    
# save the trained model to disk
torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, '../outputs/model.pth')
# plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../outputs/loss.png')
plt.show()