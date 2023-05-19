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

    def load_data(self, frequency='daily', item_type=None, type_identifier=None,
                  value_type='valoare', start_date=None, end_date=None):

        cumulator_function = getattr(self.cumstat, f'get_{frequency}_items')
        data = cumulator_function(item_type=item_type, type_identifier=type_identifier,
                                  start_date=start_date, end_date=end_date).reset_index()
        data = pd.DataFrame({'ds': data['data'], 'y': data[value_type]})
        train_split_len = int(0.8 * len(data))
        val_split_len = len(data) - train_split_len

        train_split = data.iloc[:train_split_len]
        val_split = data.iloc[train_split_len:]

        return train_split, val_split
