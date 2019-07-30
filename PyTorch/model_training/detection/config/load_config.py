import yaml


def load_yaml(config_path):
    with open(config_path, "r")as f:
        return yaml.load(f)


def get_config(config_path):
    config = load_yaml(config_path)
    return config
