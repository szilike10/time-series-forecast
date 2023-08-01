import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error

from data.data_generator import DataLoader
from forecast.model import ForecastingModel
from forecast.prophet.prophet_config import ProphetConfig
from utils.path_handling import handle_parent_path
from visualization.prophet_visualization import plot_prophet_forecast


class ProphetModel(ForecastingModel):
    def __init__(self, cfg: ProphetConfig):
        super().__init__()

        self.cfg = cfg

        self.model = Prophet(interval_width=self.cfg.interval_width,
                             daily_seasonality=self.cfg.daily_seasonality,
                             weekly_seasonality=self.cfg.weekly_seasonality,
                             yearly_seasonality=self.cfg.yearly_seasonality)

        self.train = None
        self.val = None
        self.last_trained_group = None

    def fit(self, group_name):
        self.train, self.val = self.cfg.dataloader.load_data(frequency=self.cfg.frequency,
                                                             item_type=self.cfg.group_by_col,
                                                             type_identifier=group_name,
                                                             start_date=self.cfg.start_date,
                                                             end_date=self.cfg.end_date,
                                                             min_length=self.cfg.timeseries_min_length)

        if len(self.train) + len(self.val) >= self.cfg.timeseries_min_length:
            self.model.fit(self.train)
        else:
            raise Exception(f'Not enough input data. Required: {self.cfg.timeseries_min_length}. '
                            f'Found: {len(self.train) + len(self.val)}')

        self.last_trained_group = group_name

    def predict(self, periods):
        if periods is None:
            periods = self.cfg.future_dataframe_periods

        future = self.model.make_future_dataframe(periods=periods)

        return future

    def eval(self, val=None):
        if val is None:
            val = self.val

        val_forecast = self.model.predict(val)
        loss = mean_squared_error(val['y'], val_forecast['yhat'])

        print(f'MSE: {loss}.')

        filename = f'{self.cfg.project_root}/forecast/prophet/charts/train/' \
                   f'{self.cfg.frequency}/valoare/{self.cfg.group_by_col}/{self.last_trained_group}.png'
        plot_prophet_forecast(pd.concat([self.train, self.val]),
                              val_forecast,
                              filename)

        return loss
