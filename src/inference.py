import argparse
import os

import torch
import numpy as np
from torch.utils.data import DataLoader

from my_model import resnet_model
from dataset import brain_CT_scan, valid_transforms
from visualize import plot_inference
import my_utils


def main(config_path):
    # read config
    config = my_utils.Configuration(config_path).as_dict

    # experiment_name
    exp_name = config['exp_name']

    # paths
    json_file_path_test = config['json_file_path_test']
    images_path = config['images_path']
    output_path = os.path.join(config['output_path'], exp_name)
    if not os.path.exists(output_path):
        print('there is no experiment with this name')
        return
    inference_path = os.path.join(output_path, 'inference/')
    if not os.path.exists(inference_path):
        os.makedirs(inference_path)

    # initialize the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize the model
    model = resnet_model(num_classes=15, pretrained=False, requires_grad=False).to(device)

    # load the model checkpoint
    checkpoint = torch.load(os.path.join(output_path, '/model.pth'))
    # load model weights state_dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # prepare the test dataset and dataloader
    test_data = brain_CT_scan(json_file_path_test, images_path, valid_transforms)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    for counter, data in enumerate(test_loader):
        image, target = data['image'].to(device), data['label']
        # get all the index positions where value == 1
        target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
        # get the predictions by passing the image through the model
        outputs = model(image)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.detach().cpu()
        sorted_indices = np.argsort(outputs[0])
        best = sorted_indices[-3:]
        string_predicted = ''
        string_actual = ''
        for i in range(len(best)):
            string_predicted += f"{best[i]}    "
        for i in range(len(target_indices)):
            string_actual += f"{target_indices[i]}    "

        plot_inference(image, string_predicted, string_actual, counter, inference_path)


if __name__ == '__main__':
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-cp", "--config_path", default="/misc/student/alzouabk/Thesis/supervised_multi_label_classification/configs/debugging_config.yml",
                    help="path to yaml configuration file", type=str)

    args = ap.parse_args()
