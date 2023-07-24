from pytorch_forecasting import QuantileLoss, RMSE

from config.config_base import Config


class PytorchConfig(Config):
    def __init__(self, yaml_path):
        super().__init__(yaml_path)

        self.frequency = self.yaml_obj['frequency']
        self.cumulated_csv_path = self.yaml_obj['cumulated_csv_path']
        self.smoothing_window_size = self.yaml_obj['smoothing_window_size']
        if self.cumulated_csv_path == '':
            self.cumulated_csv_path = fr'../../data/cumulated_{self.frequency}_category.csv'
        self.max_prediction_length = self.yaml_obj['max_prediction_length']
        self.max_encoder_length = self.yaml_obj['max_encoder_length']
        self.target_variable = self.yaml_obj['target_variable']
        self.group_identifiers = self.yaml_obj['group_identifiers']
        self.static_categoricals = self.yaml_obj['static_categoricals']
        self.time_varying_known_categoricals = self.yaml_obj['time_varying_known_categoricals']
        self.time_varying_known_reals = self.yaml_obj['time_varying_known_reals']
        self.time_varying_unknown_categoricals = self.yaml_obj['time_varying_unknown_categoricals']
        self.time_varying_unknown_reals = self.yaml_obj['time_varying_unknown_reals']
        self.batch_size = self.yaml_obj['batch_size']
        self.max_epochs = self.yaml_obj['max_epochs']
        self.gradient_clip = self.yaml_obj['gradient_clip']
        self.learning_rate = self.yaml_obj['learning_rate']
        self.hidden_size = self.yaml_obj['hidden_size']
        self.attention_head_size = self.yaml_obj['attention_head_size']
        self.hidden_continuous_size = self.yaml_obj['hidden_continuous_size']
        self.reduce_on_plateau_patience = self.yaml_obj['reduce_on_plateau_patience']
        self.early_stopping_patience = self.yaml_obj['early_stopping_patience']
        self.best_model_out_path = self.yaml_obj['best_model_out_path']

        loss_fn_dict = {
            'QuantileLoss': QuantileLoss(),
            'RMSE': RMSE(),
        }

        self.loss_fn = loss_fn_dict[self.yaml_obj['loss_fn']]

    def __str__(self):
        ret = ''
        for name, value in self.yaml_obj.items():
            ret += f'{name}: {value}\n'
        return ret

