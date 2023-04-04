import pandas as pd
from cumulative_statistics import CumStat


class DataLoader:
    def __init__(self, df=None, path_to_csv=None):
        if (path_to_csv and df) or (path_to_csv is None and df is None):
            raise Exception('DataLoader object should be initialized with either a path to a CSV file or a DataFrame.')
        self.cumstat = CumStat(df, path_to_csv)
        self.cumstat.add_aggregator('data', 'min')
        self.cumstat.add_aggregator('data', 'max')

    def load_data_for_article(self, cod_art, start_date=None, end_date=None):
        data = self.cumstat.cumulate_weekly_article(cod_art)
        agg = self.cumstat.group_by('cod_art')
        data = pd.DataFrame({'ds': data['data'], 'y': data['cantitate']})
        train_split_len = int(0.8 * len(data))
        val_split_len = len(data) - train_split_len

        train_split = data.iloc[:train_split_len]
        val_split = data.iloc[train_split_len:]

        return train_split, val_split
