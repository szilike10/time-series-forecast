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
    # tft_model.fit()
    path_to_model = r'C:\Users\bas6clj\time-series-forecast\forecast\pytorch\lightning_logs\lightning_logs\version_94\checkpoints\epoch=37-step=1026.ckpt'
    tft_model.eval(path_to_model=path_to_model, visualize=True)
