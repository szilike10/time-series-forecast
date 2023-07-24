import os
import pandas as pd
import lightning.pytorch as pl

from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, TemporalFusionTransformer, RMSE
from forecast.model import ForecastingModel
from forecast.pytorch.pytorch_config import PytorchConfig
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from visualization.pytorch_visualization import plot_raw_predictions


class TFTModel(ForecastingModel):
    def __init__(self, cfg: PytorchConfig):
        super().__init__()

        self.best_model_path = None
        self.trainer = None
        self.cfg = cfg

        self.data = pd.read_csv(self.cfg.cumulated_csv_path)
        # add time index
        self.data['date'] = pd.to_datetime(self.data['data'], errors='coerce')
        self.data["month"] = self.data.date.dt.month
        self.data['time_idx'] = self.data['week_no'] if self.cfg.frequency == 'weekly' \
            else self.data['date'].dt.year * 365 + self.data['date'].dt.year // 4 + self.data['date'].dt.dayofyear
        self.data["time_idx"] -= self.data["time_idx"].min()

        self.data['month'] = self.data.month.astype(str).astype("category")

        training_cutoff = self.data["time_idx"].max() - self.cfg.max_prediction_length

        self.training = TimeSeriesDataSet(
            self.data[lambda x: x.time_idx <= training_cutoff],
            time_idx="time_idx",
            target=self.cfg.target_variable,
            group_ids=self.cfg.group_identifiers,
            # keep encoder length long (as it is in the validation set)
            min_encoder_length=self.cfg.max_encoder_length // 2,
            max_encoder_length=self.cfg.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.cfg.max_prediction_length,
            static_categoricals=self.cfg.static_categoricals,
            static_reals=[],
            time_varying_known_categoricals=['month', *self.cfg.time_varying_known_categoricals],
            time_varying_known_reals=["time_idx", *self.cfg.time_varying_known_reals],
            time_varying_unknown_categoricals=[*self.cfg.time_varying_unknown_categoricals],
            time_varying_unknown_reals=[self.cfg.target_variable, *self.cfg.time_varying_unknown_reals],
            target_normalizer=GroupNormalizer(groups=self.cfg.group_identifiers, transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        self.validation = TimeSeriesDataSet.from_dataset(self.training, self.data, predict=True, stop_randomization=True)

        self.tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=self.cfg.learning_rate,
            hidden_size=self.cfg.hidden_size,
            attention_head_size=self.cfg.attention_head_size,
            dropout=0.1,
            hidden_continuous_size=self.cfg.hidden_continuous_size,
            loss=self.cfg.loss_fn,
            log_interval=10,
            # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            optimizer="Ranger",
            reduce_on_plateau_patience=self.cfg.reduce_on_plateau_patience,
        )
        print(f"Number of parameters in network: {self.tft.size() / 1e3:.1f}k")

    def fit(self):
        # create dataloaders for model
        train_dataloader = self.training.to_dataloader(train=True, batch_size=self.cfg.batch_size, num_workers=0)
        val_dataloader = self.validation.to_dataloader(train=False, batch_size=self.cfg.batch_size, num_workers=0)

        # configure network and trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=20, verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate
        logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

        self.trainer = pl.Trainer(
            max_epochs=self.cfg.max_epochs,
            accelerator="gpu",
            enable_model_summary=True,
            gradient_clip_val=0.1,
            # limit_train_batches=50,  # coment in for training, running valiation every 30 batches
            # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
            callbacks=[lr_logger, early_stop_callback],
            logger=logger,
        )

        # fit network
        self.trainer.fit(
            self.tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        self.best_model_path = self.trainer.checkpoint_callback.best_model_path

        with open(self.cfg.best_model_out_path, 'w') as f:
            f.write(self.best_model_path)

    def predict(self, x: pd.DataFrame = None, visualize=False):
        # load the best model according to the validation loss
        # (given that we use early stopping, this is not necessarily the last epoch)

        best_tft = TemporalFusionTransformer.load_from_checkpoint(self.best_model_path)

        if x is None:
            x = self.validation.to_dataloader(train=False, batch_size=self.cfg.batch_size, num_workers=0)

        # calcualte mean absolute error on validation set
        predictions = best_tft.predict(x, return_y=True, trainer_kwargs=dict(accelerator="gpu"))

        if visualize:
            img_out_prefix = f'{self.best_model_path.rsplit(os.sep, 1)[0]}/{self.cfg.frequency}'
            plot_raw_predictions(best_tft, predictions, img_out_prefix)

        return predictions

    # TODO: Add list of metrics to calculate and store them in a dictionary by name
    def eval(self, val: pd.DataFrame = None, visualize=False):
        best_tft = TemporalFusionTransformer.load_from_checkpoint(self.best_model_path)

        if val is None:
            val = self.validation.to_dataloader(train=False, batch_size=self.cfg.batch_size, num_workers=0)

        # calcualte root mean square error on validation set
        predictions = best_tft.predict(val, return_y=True, trainer_kwargs=dict(accelerator="gpu"))
        mse = RMSE()(predictions.output, predictions.y)

        print(f'RMSE = {mse}')

        if visualize:
            img_out_prefix = f'{self.best_model_path.rsplit(os.sep, 1)[0]}/{self.cfg.frequency}'
            plot_raw_predictions(best_tft, predictions, img_out_prefix)
