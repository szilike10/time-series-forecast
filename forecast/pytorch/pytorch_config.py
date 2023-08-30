import os.path

import pandas as pd
from pytorch_forecasting import QuantileLoss, RMSE

from config.config_base import Config
from data.cumulative_statistics import CumStat


class PytorchConfig(Config):
    def __init__(self, yaml_path):
        super().__init__(yaml_path)

        self.frequency = self.yaml_obj['frequency']
        self.cumulated_csv_path = self.yaml_obj.get('cumulated_csv_path', None)
        self.group_identifiers = self.yaml_obj.get('group_identifiers', None)
        self.smoothing_window_size = self.yaml_obj.get('smoothing_window_size', None)
        self.timeseries_min_length = self.yaml_obj.get('timeseries_min_length', None)
        self.start_date = self.yaml_obj.get('start_date', None)
        self.end_date = self.yaml_obj.get('end_date', None)
        group_by_list = self.group_identifiers if self.group_identifiers != ['tmp'] else []
        suffix = '' if len(group_by_list) == 0 else '_' + '_'.join(group_by_list)
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
        self.min_prediction_length = self.yaml_obj.get('min_prediction_length', None)
        self.max_prediction_length = self.yaml_obj.get('max_prediction_length', None)
        self.min_encoder_length = self.yaml_obj.get('min_encoder_length', None)
        self.max_encoder_length = self.yaml_obj.get('max_encoder_length', None)
        self.target_variable = self.yaml_obj.get('target_variable', None)
        self.static_categoricals = self.yaml_obj.get('static_categoricals', None)
        self.time_varying_known_categoricals = self.yaml_obj.get('time_varying_known_categoricals', None)
        self.time_varying_known_reals = self.yaml_obj.get('time_varying_known_reals', None)
        self.time_varying_unknown_categoricals = self.yaml_obj.get('time_varying_unknown_categoricals', None)
        self.time_varying_unknown_reals = self.yaml_obj.get('time_varying_unknown_reals', None)
        self.device = self.yaml_obj.get('device', None)
        self.batch_size = self.yaml_obj.get('batch_size', None)
        self.max_epochs = self.yaml_obj.get('max_epochs', None)
        self.gradient_clip = self.yaml_obj.get('gradient_clip', None)
        self.learning_rate = self.yaml_obj.get('learning_rate', None)
        self.hidden_size = self.yaml_obj.get('hidden_size', None)
        self.lstm_layers = self.yaml_obj.get('lstm_layers', None)
        self.attention_head_size = self.yaml_obj.get('attention_head_size', None)
        self.hidden_continuous_size = self.yaml_obj.get('hidden_continuous_size', None)
        self.dropout = self.yaml_obj['dropout']
        self.reduce_on_plateau_patience = self.yaml_obj.get('reduce_on_plateau_patience', None)
        self.early_stopping_patience = self.yaml_obj.get('early_stopping_patience', None)
        self.best_model_out_path = self.yaml_obj.get('best_model_out_path', None)
        self.start_date = pd.to_datetime(self.yaml_obj.get('start_date', None))
        self.end_date = pd.to_datetime(self.yaml_obj.get('end_date', None))

        self.timeseries_length = (self.end_date - self.start_date).days

        loss_fn_dict = {
            'QuantileLoss': QuantileLoss(quantiles=[0.05, 0.1, 0.5, 0.9, 0.95]),
            'RMSE': RMSE(),
        }

        self.loss_fn = loss_fn_dict[self.yaml_obj.get('loss_fn', None)]

    def __str__(self):
        ret = ''
        for name, value in self.yaml_obj.items():
            ret += f'{name}: {value}\n'
        return ret

