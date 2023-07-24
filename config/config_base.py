import yaml
from yaml.loader import SafeLoader


class Config:
    def __init__(self, yaml_path):
        self.yaml_path = yaml_path

        with open(yaml_path, 'r') as f:
            self.yaml_obj = yaml.load(f, SafeLoader)

        pass

if __name__ == '__main__':
    cfg = Config(r'C:\Users\bas6clj\time-series-forecast\forecast\pytorch\pytorch_config.yml')

    pass