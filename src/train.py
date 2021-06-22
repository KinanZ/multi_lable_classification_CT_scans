import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from engine import train, validate
from dataset import brain_CT_scan, train_transforms, valid_transforms
from my_model import resnet_model
import my_utils


def main(config_path):
    # read config
    config = my_utils.Configuration(config_path).as_dict

    # paths
    json_file_path = config['json_file_path']
    images_path = config['images_path']

    # seed
    random_seed = config['SEED']

    train_dataset = brain_CT_scan(json_file_path, images_path, train_transforms)
    valid_dataset = brain_CT_scan(json_file_path, images_path, valid_transforms)

    # data loading parameters
    validation_split = config['validation_split']
    shuffle_dataset = config['shuffle_dataset']
    batch_size = config['batch_size']

    # Creating data indices for training and validation splits:
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize the model
    model = resnet_model(num_classes=15, pretrained=config['pretrained'], requires_grad=config['requires_grad']).to(device)

    # learning parameters
    lr = config['lr']
    epochs = config['epochs']
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
            model, validation_loader, criterion, valid_dataset, device
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
    plt.plot(valid_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../outputs/loss.png')


if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-cp", "--config_path", default="../configs/debugging_config.yml",
                    help="path to yaml configuration file", type=str)

    args = ap.parse_args()

main(config_path=args.config_path)
