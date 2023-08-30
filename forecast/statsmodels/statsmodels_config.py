import os

import pandas as pd

from config.config_base import Config
from data.cumulative_statistics import CumStat


class StatsmodelsConfig(Config):
    def __init__(self, yaml_path):
        super().__init__(yaml_path)

        self.combined_csv_path = self.yaml_obj.get('combined_csv_path', None)
        if self.combined_csv_path is None:
            self.combined_csv_path = fr'{self.project_root}/data/combined.csv'
        self.frequency = self.yaml_obj.get('frequency', None)
        self.group_identifiers = self.yaml_obj.get('group_identifiers', None)
        self.type_identifier = self.yaml_obj.get('type_identifier', None)
        self.target = self.yaml_obj.get('target', None)
        self.smoothing_window_size = self.yaml_obj.get('smoothing_window_size', None)
        self.timeseries_min_length = self.yaml_obj['timeseries_min_length']
        self.conf_alpha = self.yaml_obj.get('conf_alpha', None)
        self.start_date = pd.to_datetime(self.yaml_obj.get('start_date', None))
        self.end_date = pd.to_datetime(self.yaml_obj.get('end_date', None))
        group_by_list = self.group_identifiers if self.group_identifiers != ['tmp'] else []
        suffix = '' if len(group_by_list) == 0 else '_' + '_'.join(group_by_list)
        self.cumulated_csv_path = self.yaml_obj.get('cumulated_csv_path', '')
        if self.cumulated_csv_path == '':
            self.cumulated_csv_path = fr'{self.project_root}/data/cumulated_{self.frequency}{suffix}.csv'
        if not os.path.exists(self.cumulated_csv_path):
            cumstat = CumStat(path_to_csv=f'{self.project_root}/data/combined.csv')
            cum_func = getattr(cumstat, f'cumulate_{self.frequency}')
            # group_by_col = 'category' if 'category' in self.group_identifiers else 'cod_art'
            cum_func(group_by_column=self.group_identifiers if self.group_identifiers != ['tmp'] else None,
                     filter_under=self.timeseries_min_length,
                     start_date=self.start_date,
                     end_date=self.end_date)

