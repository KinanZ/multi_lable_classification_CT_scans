import os
import yaml


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