import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


if __name__ == '__main__':
    data = pd.read_csv('../../data/cumulated_weekly_category.csv')

    # add time index
    data['date'] = pd.to_datetime(data['data'], errors='coerce')
    # data["time_idx"] = data["date"].dt.year * 365 + data["date"].dt.year // 4 + data["date"].dt.dayofyear
    data["month"] = data.date.dt.month
    data["time_idx"] = data["week_no"]
    # data['time_idx'] = data['time_idx'] - 52 * data[]
    data["time_idx"] -= data["time_idx"].min()

    # add additional features
    data['month'] = data.month.astype(str).astype("category")  # categories have be strings

    # we want to encode special days as one variable and thus need to first reverse one-hot encoding
    # special_days = [
    #     "easter_day",
    #     "good_friday",
    #     "new_year",
    #     "christmas",
    #     "labor_day",
    #     "independence_day",
    #     "revolution_day_memorial",
    #     "regional_games",
    #     "fifa_u_17_world_cup",
    #     "football_gold_cup",
    #     "beer_capital",
    #     "music_fest",
    # ]
    # data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
    # data.sample(10, random_state=521)

    data.describe()

    max_prediction_length = 4
    max_encoder_length = 40
    training_cutoff = data["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="valoare",
        group_ids=['category'],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=['category'],
        static_reals=[],
        time_varying_known_categoricals=['month'],
        time_varying_known_reals=["time_idx", 'pret'],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            'valoare'
        ],
        target_normalizer=GroupNormalizer(
            groups=['category'], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    # create dataloaders for model
    batch_size = 64  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        # callbacks=[lr_logger, early_stop_callback],
        # logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # load the best model according to the validation loss
    # (given that we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # calcualte mean absolute error on validation set
    predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
    MAE()(predictions.output, predictions.y)

    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)

    print(len(val_dataloader))


    for idx in range(len(raw_predictions.x)):  # plot 10 examples
        best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)

        plt.savefig(f'weekly_{idx}.png', dpi=600)
