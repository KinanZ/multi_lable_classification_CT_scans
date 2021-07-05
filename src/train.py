import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from engine import train, validate
from dataset import brain_CT_scan, train_transforms, valid_transforms
from my_model import resnet_model
from visualize import plot_loss, plot_evaluation_metrics
import my_utils


def main(config_path):
    # read config
    config = my_utils.Configuration(config_path).as_dict

    # experiment_name
    exp_name = config['exp_name']

    # paths
    json_file_path_train = config['json_file_path_train']
    json_file_path_test = config['json_file_path_test']
    images_path = config['images_path']
    output_path = os.path.join(config['output_path'], exp_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    train_dataset = brain_CT_scan(json_file_path_train, images_path, train_transforms)
    valid_dataset = brain_CT_scan(json_file_path_test, images_path, valid_transforms)

    # data loading parameters
    shuffle_dataset = config['shuffle_dataset']
    batch_size = config['batch_size']

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

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
    eval_results = {'accuracy': [], 'micro/precision': [], 'micro/recall': [], 'micro/f1': [],
                    'macro/precision': [], 'macro/recall': [], 'macro/f1': [],
                    'sample/precision': [], 'sample/recall': [], 'sample/f1': []}
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = train(
            model, train_loader, optimizer, criterion, train_dataset, device
        )
        if config['evaluate']:
            valid_epoch_loss, results = validate(
                model, validation_loader, criterion, valid_dataset, device
            )
            for key in eval_results:
                eval_results[key].extend(results[key])
        else:
            valid_epoch_loss = validate(
                model, validation_loader, criterion, valid_dataset, device
            )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f'Val Loss: {valid_epoch_loss:.4f}')
        if config['evaluate']:
            print("Accuracy: {:.3f}"
                  "micro f1: {:.3f} "
                  "macro f1: {:.3f} "
                  "samples f1: {:.3f} ".format(results['accuracy'],
                                               results['micro/f1'],
                                               results['macro/f1'],
                                               results['samples/f1'],
                                               ))

    # save the trained model to disk
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(output_path, '/model.pth'))

    # plot and save the train and validation line graphs
    if config['plot_curves']:
        plot_loss(train_loss, valid_loss, output_path)
        plot_evaluation_metrics(eval_results, output_path)


if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-cp", "--config_path", default="/misc/student/alzouabk/Thesis/supervised_multi_label_classification/configs/debugging_config.yml",
                    help="path to yaml configuration file", type=str)

    args = ap.parse_args()

main(config_path=args.config_path)
