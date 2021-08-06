import os
import yaml
import csv
import numpy as np
import torch
import elasticdeform.torch as etorch

def read_yaml(yaml_path):
    """
    Reads yaml file as dict.
    """
    with open(yaml_path) as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    return yaml_data


def save_yaml(dict_file, yaml_path):
    """
    Saves a dict as yaml file.
    """
    with open(yaml_path, "w") as file:
        documents = yaml.dump(dict_file, file)


class Configuration:
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    default_config_path = os.path.join(current_file_path, "../configs", "default_config.yml")

    def __init__(self, config_path: str = None, default_config_path=default_config_path):
        # read base config
        base_config = read_yaml(default_config_path)
        if config_path is not None:
            config = read_yaml(config_path)
            # overwrite base config
            base_config.update(config)
        # set overwritten config
        self.as_dict = base_config


def save_csv(data, path):
    header = data.keys()
    no_rows = len(data[list(header)[0]])

    with open(os.path.join(path, 'eval_metrics.csv'), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(header)
        for row in range(no_rows):
            csvwriter.writerow([data[key][row] for key in header])


def elastic_deform(x, control_points_num=3, sigma=20, axis=(1, 2)):
    # generate a deformation grid
    displacement = np.random.randn(2, control_points_num, control_points_num) * sigma
    # construct PyTorch input and top gradient
    displacement = torch.tensor(displacement)
    # elastic deformation
    ed_x = etorch.deform_grid(x, displacement, prefilter=True, axis=axis)
    return ed_x
