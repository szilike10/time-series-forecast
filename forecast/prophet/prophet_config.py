import pandas as pd
from config.config_base import Config
from data.data_generator import DataLoader


class ProphetConfig(Config):
    def __init__(self, yaml_path):
        super().__init__(yaml_path)

        self.combined_csv_path = self.yaml_obj['combined_csv_path']
        if self.combined_csv_path is None :
            self.combined_csv_path = fr'{self.project_root}/data/combined.csv'
        self.frequency = self.yaml_obj['frequency']
        self.group_identifiers = self.yaml_obj['group_identifiers']
        self.smoothing_window_size = self.yaml_obj['smoothing_window_size']
        self.daily_seasonality = self.yaml_obj['daily_seasonality']
        self.weekly_seasonality = self.yaml_obj['weekly_seasonality']
        self.yearly_seasonality = self.yaml_obj['yearly_seasonality']
        self.interval_width = self.yaml_obj['interval_width']
        self.future_dataframe_periods = self.yaml_obj['future_dataframe_periods']
        self.timeseries_min_length = self.yaml_obj['timeseries_min_length']
        self.start_date = pd.to_datetime(self.yaml_obj['start_date'])
        self.end_date = pd.to_datetime(self.yaml_obj['end_date'])
        self.group_by_col = 'category' if 'category' in self.group_identifiers else 'cod_art'
        self.dataloader = DataLoader(path_to_csv=self.combined_csv_path)