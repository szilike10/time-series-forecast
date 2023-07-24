import argparse

from forecast.pytorch.model import TFTModel
from pytorch_config import PytorchConfig


def load_config(path):
    return PytorchConfig(path)


def create_model(cfg):
    return TFTModel(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str)
    args = parser.parse_args()

    cfg_path = args.cfg_path

    cfg = load_config(cfg_path)
    tft_model = create_model(cfg)
    tft_model.fit()
    tft_model.eval()
