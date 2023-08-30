import argparse

import pandas as pd

from forecast.statsmodels.model import StatModel
from forecast.statsmodels.statsmodels_config import StatsmodelsConfig


def load_config(path):
    return StatsmodelsConfig(path)


def create_model(cfg):
    return StatModel(cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str)
    args = parser.parse_args()

    cfg_path = args.cfg_path

    cfg = load_config(cfg_path)
    statmodel_model = create_model(cfg)

    # model = statmodel_model.fit()
    statmodel_model.eval()
