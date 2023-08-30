import argparse

from forecast.prophet.model import ProphetModel
from forecast.prophet.prophet_config import ProphetConfig


def load_config(path):
    return ProphetConfig(path)


def create_model(cfg):
    return ProphetModel(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str)
    args = parser.parse_args()

    cfg_path = args.cfg_path

    cfg = load_config(cfg_path)
    prophet_model = create_model(cfg)
    prophet_model.fit()
    prophet_model.eval()
