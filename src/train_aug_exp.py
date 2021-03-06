import os
import argparse
import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from engine import train, validate
from dataset import brain_CT_scan, MultilabelBalancedRandomSampler
from my_model import resnet_model
from visualize import plot_loss, plot_evaluation_metrics
import my_utils


def main(config_path):
    # starting time:
    start_time = time.time()

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

    # save config file to exp folder
    my_utils.save_yaml(config, os.path.join(output_path, 'exp_config.yml'))

    # logging config
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(output_path, 'out.log'),
                        format='%(asctime)s :: %(levelname)s :: %(message)s')

    # get if input is stacked multi-channel or not
    stack_pre_post = config['stack_pre_post']

    # Augmentations
    # Normalize mean and std
    normalize = transforms.Normalize(mean=[config['Normalize_mean'], config['Normalize_mean'], config['Normalize_mean']],
                                         std=[config['Normalize_std'], config['Normalize_std'], config['Normalize_std']])
    axis = (1, 2)

    # train transforms as setup by config.yml
    train_transforms = transforms.Compose([
        transforms.RandomApply([
            transforms.Lambda(lambda x: my_utils.elastic_deform(x,
                                                                control_points_num=config[
                                                                    'elasticdeform_control_points_num'],
                                                                sigma=config['elasticdeform_sigma'], axis=axis))
        ], p=config['elasticdeform_p']),
        transforms.RandomApply([
            transforms.Lambda(lambda x: my_utils.elastic_deform(x,
                                                                control_points_num=config['elasticdeform_control_points_num'],
                                                                sigma=config['elasticdeform_sigma'], axis=axis))
        ], p=config['elasticdeform_p']),
        transforms.RandomApply([
            transforms.RandomResizedCrop(config['RandomResizedCrop_size'], scale=config['RandomResizedCrop_scale'])
        ], p=config['RandomResizedCrop_p']),
        transforms.RandomApply([
            transforms.RandomRotation(config['RandomRotation_range'])
        ], p=config['RandomRotation_p']),
        transforms.RandomApply([
            transforms.RandomAffine(config['RandomAffine_rotate'], translate=config['RandomAffine_translate'], scale=config['RandomAffine_scale']
                                    , shear=config['RandomAffine_shear'])
        ], p=config['RandomAffine_p']),
        transforms.RandomApply([
            transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, brightness_factor=config['adjust_brightness_factor']))
        ], p=config['adjust_brightness_p']),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=config['GaussianBlur_kernel_size'], sigma=config['GaussianBlur_sigma'])
        ], p=config['GaussianBlur_p']),
        transforms.RandomErasing(p=config['RandomErasing_p'], scale=config['RandomErasing_scale'],
                                 ratio=config['RandomErasing_ratio'], value=0, inplace=False),
        transforms.RandomHorizontalFlip(p=config['RandomHorizontalFlip_p']),
        transforms.RandomApply([normalize], p=config['Normalize_p']),
    ])

    # valid transforms
    valid_transforms = transforms.Compose([
        transforms.RandomApply([normalize], p=config['Normalize_p']),
    ])

    if config['bbox_aug']:
        train_dataset = brain_CT_scan(json_file_path_train, images_path, train_transforms, stack_pre_post, bbox_aug= True)
    else:
        train_dataset = brain_CT_scan(json_file_path_train, images_path, train_transforms, stack_pre_post)
    valid_dataset = brain_CT_scan(json_file_path_test, images_path, valid_transforms, stack_pre_post)

    # data loading parameters
    train_sampler = MultilabelBalancedRandomSampler(train_dataset.y, class_choice="least_sampled")
    batch_size = config['batch_size']

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize the model
    model = resnet_model(num_classes=15, pretrained=config['pretrained'],
                         requires_grad=config['requires_grad']).to(device)

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
                    'samples/precision': [], 'samples/recall': [], 'samples/f1': []}
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1} of {epochs}")
        logging.info("Training..")
        train_epoch_loss = train(
            model, train_loader, optimizer, criterion, train_dataset, device
        )
        logging.info("Validation..")
        if config['evaluate']:
            valid_epoch_loss, results = validate(
                model, validation_loader, criterion, valid_dataset, device, evaluate=True)
            for key in eval_results:
                eval_results[key].append(results[key])
        else:
            valid_epoch_loss = validate(
                model, validation_loader, criterion, valid_dataset, device
            )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        logging.info(f"Train Loss: {train_epoch_loss:.4f}")
        logging.info(f'Val Loss: {valid_epoch_loss:.4f}')
        if config['evaluate']:
            logging.info("Accuracy: {:.3f} "
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
                }, os.path.join(output_path, 'model.pth'))

    # save and/or plot the train and validation line graphs
    if config['plot_curves']:
        plot_loss(train_loss, valid_loss, output_path)
        if config['evaluate']:
            my_utils.save_csv(eval_results, output_path)
            plot_evaluation_metrics(eval_results, output_path)

    # time at the end and print it:
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logging.info("Training is done!, Execution time was: ")
    logging.info("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-cp", "--config_path", default="/misc/student/alzouabk/Thesis/supervised_multi_label_classification/configs/debugging_config.yml",
                    help="path to yaml configuration file", type=str)

    args = ap.parse_args()

main(config_path=args.config_path)
