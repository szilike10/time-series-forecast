import pandas as pd
from data.cumulative_statistics import CumStat


class DataLoader:
    def __init__(self, df=None, path_to_csv=None):
        if (path_to_csv and df) or (path_to_csv is None and df is None):
            raise Exception('DataLoader object should be initialized with either a path to a CSV file or a DataFrame.')
        self.cumstat = CumStat(df=df, path_to_csv=path_to_csv)
        self.cumulated_data = None

    def get_article_ids(self):
        return self.cumstat.get_all_article_ids()

    def get_categories(self):
        return self.cumstat.get_all_categories()

    def load_data_for_article(self, cod_art, fill_missing_data=False, start_date=None, end_date=None):
        self.cumstat.add_aggregator('data', 'min')
        self.cumstat.add_aggregator('data', 'max')
        # if self.cumulated_data is None:
        data = self.cumstat.cumulate_daily_article(cod_art, fill_missing_data)
        data = pd.DataFrame({'ds': data['data'], 'y': data['cantitate']})
        train_split_len = int(0.8 * len(data))
        val_split_len = len(data) - train_split_len

        train_split = data.iloc[:train_split_len]
        val_split = data.iloc[train_split_len:]

        return train_split, val_split

    def load_data_for_category(self, category, fill_missing_data=False, start_date=None, end_date=None):
        self.cumstat.add_aggregator('data', 'min')
        self.cumstat.add_aggregator('data', 'max')
        # if self.cumulated_data is None:
        data = self.cumstat.cumulate_daily_category(category, fill_missing_data)
        data = pd.DataFrame({'ds': data['data'], 'y': data['cantitate']})
        train_split_len = int(0.8 * len(data))
        val_split_len = len(data) - train_split_len

        train_split = data.iloc[:train_split_len]
        val_split = data.iloc[train_split_len:]

        return train_split, val_split

    def load_combined_data(self, column='valoare', fill_missing_data=False, start_date=None, end_date=None):
        self.cumstat.add_aggregator('data', 'min')
        self.cumstat.add_aggregator('data', 'max')
        # if self.cumulated_data is None:
        data = self.cumstat.cumulate_daily_all_articles()
        data = pd.DataFrame({'ds': data['data'], 'y': data[column]})
        train_split_len = int(0.8 * len(data))
        val_split_len = len(data) - train_split_len

        train_split = data.iloc[:train_split_len]
        val_split = data.iloc[train_split_len:]

        return train_split, val_split