import pandas as pd
from config.config_base import Config
from data.data_generator import DataLoader


class ProphetConfig(Config):
    def __init__(self, yaml_path):
        super().__init__(yaml_path)

        self.combined_csv_path = self.yaml_obj.get('combined_csv_path', None)
        if self.combined_csv_path is None :
            self.combined_csv_path = fr'{self.project_root}/data/combined.csv'
        self.frequency = self.yaml_obj.get('frequency', None)
        self.group_identifiers = self.yaml_obj.get('group_identifiers', None)
        self.smoothing_window_size = self.yaml_obj.get('smoothing_window_size', None)
        self.daily_seasonality = self.yaml_obj.get('daily_seasonality', None)
        self.weekly_seasonality = self.yaml_obj.get('weekly_seasonality', None)
        self.yearly_seasonality = self.yaml_obj.get('yearly_seasonality', None)
        self.interval_width = self.yaml_obj.get('interval_width', None)
        self.future_dataframe_periods = self.yaml_obj.get('future_dataframe_periods', None)
        self.timeseries_min_length = self.yaml_obj.get('timeseries_min_length', None)
        self.start_date = pd.to_datetime(self.yaml_obj.get('start_date', None))
        self.end_date = pd.to_datetime(self.yaml_obj.get('end_date', None))
        if 'category' in self.group_identifiers:
            self.group_by_col = 'category'
        elif 'cod_art' in self.group_identifiers:
            self.group_by_col = 'cod_art'
        elif 'cluster' in self.group_identifiers:
            self.group_by_col = 'cluster'
        else:
            self.group_by_col = None
        self.dataloader = DataLoader(path_to_csv=self.combined_csv_path)
