import os
import random
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


def clip(box):
    new_box = (max(min(int(round(int(round(box[0])))), 512), 0),
               max(min(int(round(int(round(box[1])))), 512), 0))
    return new_box


def stretch(bbox, factor=.2):
    # Arguments:
    bbox2 = []
    for dim in ((bbox[0], bbox[2]), (bbox[1], bbox[3])):
        cur_min, cur_max = dim
        rnd_min, rnd_max = clip((cur_min - np.random.chisquare(df=3) / 8 * cur_min,
                                 cur_max + np.random.chisquare(df=3) / 8 * (512 - cur_max)))

        bbox2.append((rnd_min, rnd_max))
    return (bbox2[0][0], bbox2[1][0], bbox2[0][1], bbox2[1][1])


def crop_show_augment(image, labels, bboxes):
    # show the diseased areas based on bounding boxes
    tmp = np.zeros((512, 512, 3), dtype=np.uint8)
    if labels[0] == 1:
        bboxes = random.sample(range(48, 464), 2)
        bboxes.append(random.randint(bboxes[0], 464))
        bboxes.append(random.randint(bboxes[1], 464))
    for b in bboxes:
        print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb: ', b)
        b = stretch(b)
        tmp[b[1]:b[3], b[0]:b[2], :] = np.asarray(image)[b[1]:b[3], b[0]:b[2], :]
    return tmp
