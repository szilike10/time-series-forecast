import os.path
from copy import deepcopy

import pandas as pd
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np

from utils.data_processing import fill_missing_dates


class CumStat:
    """
    A class to quickly apply aggregations on the given dataframe.
    """

    def __init__(self, df=None, path_to_csv=None, ):
        if (path_to_csv and df) or (path_to_csv is None and df is None):
            raise Exception('CumStat object should be initialized with either a path to a CSV file or a DataFrame.')
        self.path_to_csv = None
        if path_to_csv:
            self.path_to_csv = path_to_csv
            self.df = pd.read_csv(path_to_csv)

        else:
            self.df = df
        self.df['data'] = pd.to_datetime(self.df['data'])
        self.df['year'] = self.df['data'].dt.year
        self.df['month'] = self.df['data'].dt.month
        self.aggregators = []

    def get_all_article_ids(self):
        return np.sort(self.df['cod_art'].unique())

    def get_all_categories(self):
        return np.sort(self.df['category'].unique())

    def _fill_df_with_day(self, df):
        df['day'] = df['data']

        return df

    def _fill_df_with_week(self, df):
        df['week'] = df['data'].dt.to_period('W').dt.start_time + pd.Timedelta(6, unit='d')
        df = self._add_weekno(df)

        return df

    def _add_weekno(self, df):
        df['week_no_annual'] = df['data'].dt.isocalendar().week
        year_adjusted = (df['month'] == 1).astype(int).multiply((df['week_no_annual'] > 50).astype(int))
        df['week_no'] = (df['year'] - year_adjusted) * 52 + df['week_no_annual']
        df['week_no'] -= df['week_no'].min()

        return df

    def preprocess_data(self, df, col, filter_under=0):
        ret_df = pd.DataFrame()

        for name in df[col].unique():
            current = df[df[col] == name]
            if len(current) > filter_under:
                filled = fill_missing_dates(current, 'data', ['cantitate', 'valoare'])
                ret_df = pd.concat([ret_df, filled])

        return ret_df

    def add_aggregator(self, column_name: str, aggregator_function: str):

        """
        Adds an aggregator function to a given column name.

        :param column_name: name of the aggregated column
        :param aggregator_function: the function that we
               want to apply on the elements of the column
        """

        self.aggregators.append((column_name, aggregator_function))

    def group_by(self, group_by):

        """
        Applies group by for the selected field and aggregates the stored functions.

        :param group_by: the column in the dataframe to apply the group by to
        :return: a grouped dataframe containing the aggregations
        """

        return self.df.groupby(group_by) \
            .agg(**{f'{agg[1]}_{agg[0]}': (agg[0], agg[1]) for agg in self.aggregators}).reset_index()

    def cumulate_weekly(self, group_by_column=None, start_date=None, end_date=None, filter_under=0):

        """
        Helper function to create the weekly cumulated DataFrame.
        """

        group_by_list = []
        if group_by_column is not None:
            if type(group_by_column) is list:
                group_by_list.extend(group_by_column)
            else:
                group_by_list.append(group_by_column)

        path_to_cached_df = f'{os.environ["PROJECT_ROOT"]}/data/cumulated_weekly_{"_".join(group_by_list[:-1])}.csv'
        if os.path.exists(path_to_cached_df):
            return pd.read_csv(path_to_cached_df)

        ret_df = deepcopy(self.df)
        if len(group_by_list) > 1:
            new_col = '_'.join(group_by_list)
            ret_df[new_col] = ret_df[group_by_list[0]]
            for col_name in group_by_list[1:]:
                ret_df[new_col] += '_' + ret_df[col_name].map(str)
        else:
            new_col = 'data'

        ret_df = self.preprocess_data(ret_df, new_col, filter_under)
        ret_df = self._fill_df_with_week(ret_df)
        group_by_list.append('week_no')
        ret_df = ret_df.sort_values(by=group_by_list)
        ret_df = ret_df.groupby(group_by_list).agg(cantitate=('cantitate', 'sum'),
                                                    pret=('pret', 'mean'),
                                                    valoare=('valoare', 'sum'),
                                                    week=('week', 'first')).reset_index()

        ret_df['data'] = ret_df['week']
        if start_date:
            ret_df = ret_df.query('data >= @start_date')
        if end_date:
            ret_df = ret_df.query('data <= @end_date')

        if new_col != 'data' and filter_under > 0:
            counted = ret_df.groupby(new_col, as_index=False).size()
            for i in range(len(counted)):
                count = int(counted['size'][i])
                key = counted.iloc[i][new_col]
                if count < filter_under:
                    ret_df.drop(ret_df.loc[ret_df[new_col] == key].index, inplace=True)

        ret_df = ret_df.reset_index(drop=True)

        ret_df.to_csv(path_to_cached_df, index=True)

        return ret_df

    def cumulate_daily(self, group_by_column=None, start_date=None, end_date=None, filter_under=0):

        """
        Helper function to create the daily cumulated DataFrame.
        """

        group_by_list = []
        if group_by_column is not None:
            if type(group_by_column) is list:
                group_by_list.extend(group_by_column)
            else:
                group_by_list.append(group_by_column)

        path_to_cached_df = f'{os.environ["PROJECT_ROOT"]}/data/cumulated_daily_{"_".join(group_by_list[:-1])}.csv'
        if os.path.exists(path_to_cached_df):
            return pd.read_csv(path_to_cached_df)

        ret_df = deepcopy(self.df)
        if len(group_by_list) > 1:
            new_col = '_'.join(group_by_list)
            ret_df[new_col] = ret_df[group_by_list[0]]
            for col_name in group_by_list[1:]:
                ret_df[new_col] += '_' + ret_df[col_name].map(str)
        else:
            new_col = 'data'

        ret_df = self.preprocess_data(ret_df, new_col, filter_under)
        ret_df = self._fill_df_with_day(ret_df)
        group_by_list.append('day')
        ret_df = ret_df.sort_values(by=group_by_list)
        ret_df = ret_df.groupby(group_by_list).agg(cantitate=('cantitate', 'sum'),
                                                    pret=('pret', 'mean'),
                                                    valoare=('valoare', 'sum'),
                                                    um=('um', 'first')).reset_index()
        ret_df['data'] = ret_df['day']

        if start_date:
            ret_df = ret_df.query('data >= @start_date')
        if end_date:
            ret_df = ret_df.query('data <= @end_date')

        # TODO: correct filtering
        if new_col != 'data' and filter_under > 0:
            new_col = '-'.join(group_by_list)
            tmp_df = ret_df
            counted = ret_df.groupby(new_col, as_index=False).size()
            for i in range(len(counted)):
                count = int(counted['size'][i])
                key = counted.iloc[i][new_col]
                if count < filter_under:
                    ret_df.drop(ret_df.loc[ret_df[new_col] == key].index, inplace=True)
        #
        # for unique_name in ret_df[new_col].unique():
        #     idx = ret_df[ret_df[new_col] == unique_name].index.values
        #     max_valoare = ret_df[ret_df[new_col] == unique_name].max()['valoare']
        #     max_cantitate = ret_df[ret_df[new_col] == unique_name].max()['cantitate']
        #     ret_df.loc[idx, 'valoare'] /= max_valoare / 10
        #     ret_df.loc[idx, 'cantitate'] /= max_cantitate / 10

        ret_df = ret_df.reset_index(drop=True)

        ret_df.to_csv(path_to_cached_df, index=True)

        return ret_df

    def get_daily_items(self, item_type=None, type_identifier=None, fill_missing_dates=False,
                        start_date=None, end_date=None, filter_under=0):
        if (item_type, type_identifier) == (None, not None) \
                or (item_type, type_identifier) == (not None, None):
            raise Exception('item_type and type identifier should either be both specfied or none of them')

        ret_df = self.cumulate_daily(group_by_column=item_type,
                                     start_date=start_date,
                                     end_date=end_date,
                                     filter_under=filter_under)

        if item_type is not None:
            ret_df = ret_df.query(f'{item_type} == @type_identifier')

        if not fill_missing_dates:
            return ret_df

    def get_weekly_items(self, item_type=None, type_identifier=None, fill_missing_dates=False,
                         start_date=None, end_date=None, filter_under=0):
        if (item_type, type_identifier) == (None, not None) \
                or (item_type, type_identifier) == (not None, None):
            raise Exception('item_type and type identifier should either be both specfied or none of them')

        ret_df = self.cumulate_weekly(group_by_column=item_type,
                                      start_date=start_date,
                                      end_date=end_date,
                                      filter_under=filter_under)

        if item_type is not None:
            ret_df = ret_df.query(f'{item_type} == @type_identifier')

        if not fill_missing_dates:
            return ret_df

    def plot_article_weekly(self, cod_art, filename=None):
        weekly_df = self.cumulate_weekly(group_by_column='cod_art').query('cod_art == @cod_art')
        weekly_df.plot(y='cantitate', use_index=True)

        plt.xticks(rotation=60)
        plt.xticks(fontsize=10)
        plt.ylabel('Cantitate')
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_article_daily(self, cod_art, filename=None):
        daily_df = self.cumulate_daily(group_by_column='cod_art').query('cod_art == @cod_art')
        daily_df.plot(y='cantitate', use_index=True)

        plt.xticks(rotation=60)
        plt.xticks(fontsize=10)
        plt.ylabel('Cantitate')
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_category_daily(self, category, filename=None):
        daily_df = self.cumulate_daily(group_by_column='category').query('category == @category')
        filter_count = 50
        if len(daily_df) < filter_count:
            print(f'For category {category} there were less than {filter_count} values found.')
            return
        daily_df.plot(y='cantitate', use_index=True)

        plt.xticks(rotation=60)
        plt.xticks(fontsize=10)
        plt.ylabel('Cantitate')
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_daily_combined(self, column='cantitate', filename=None):
        daily_df = self.cumulate_daily(start_date='2022-01-01', end_date='2023-01-01')

        daily_df.plot(y=column, use_index=True)

        plt.xticks(rotation=60)
        plt.xticks(fontsize=10)
        plt.ylabel(column)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_weekly_combined(self, column='cantitate', filename=None):
        weekly_df = self.cumulate_weekly(start_date='2022-01-01', end_date='2023-01-01')

        weekly_df.plot(y=column, use_index=True)

        plt.xticks(rotation=60)
        plt.xticks(fontsize=10)
        plt.ylabel(column)
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()


def test_features(path_to_csv):
    path_to_csv = args.path_to_csv

    cumstat = CumStat(path_to_csv=path_to_csv)
    cumstat.add_aggregator('data', 'min')
    cumstat.add_aggregator('data', 'max')
    cumstat.add_aggregator('cod_art', 'count')
    cumstat.add_aggregator('cantitate', 'min')
    cumstat.add_aggregator('cantitate', 'max')
    cumstat.add_aggregator('cantitate', 'mean')
    cumstat.add_aggregator('cantitate', 'std')

    agg = cumstat.group_by('cod_art').reset_index()
    print(agg.to_string())

    weekly_cumstat = CumStat(df=agg)
    weekly_cumstat.add_aggregator('cantitate', 'min')
    weekly_cumstat.add_aggregator('cantitate', 'max')
    weekly_cumstat.add_aggregator('cantitate', 'mean')
    weekly_cumstat.add_aggregator('cantitate', 'std')

    weekly_agg = weekly_cumstat.group_by('cod_art').reset_index()
    print(weekly_agg)

    cumstat.plot_article_weekly(74, 'weekly_74.png')


def plot_all_articles(fill_missing_articles=False):
    cumstat = CumStat(path_to_csv=path_to_csv)
    cumstat.add_aggregator('cod_art', 'count')
    agg = cumstat.group_by('cod_art').reset_index()
    for cod_art in agg['cod_art']:
        cumstat.plot_article_daily(cod_art, f'all_articles/filled/daily_{cod_art}.png')


def plot_all_categories(fill_missing_articles=False):
    cumstat = CumStat(path_to_csv=path_to_csv)
    categories = cumstat.get_all_categories()
    for category in categories:
        cumstat.plot_category_daily(category, f'all_categories/not_filled/daily_{category}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv', type=str)
    parser.add_argument('--monthly_stat_out_path', type=str, required=False)
    parser.add_argument('--weekly_stat_out_path', type=str, required=False)
    parser.add_argument('--monthly_df', type=str, required=False)
    parser.add_argument('--weekly_df', type=str, required=False)
    args = parser.parse_args()

    path_to_csv = args.path_to_csv
    monthly_stat_out_path = args.monthly_stat_out_path
    weekly_stat_out_path = args.weekly_stat_out_path
    monthly_df = args.monthly_df
    weekly_df = args.weekly_df

    cumstat = CumStat(path_to_csv=path_to_csv)

    cumstat.add_aggregator('data', 'min')
    cumstat.add_aggregator('data', 'max')
    cumstat.add_aggregator('cod_art', 'count')
    cumstat.add_aggregator('cantitate', 'min')
    cumstat.add_aggregator('cantitate', 'max')
    cumstat.add_aggregator('cantitate', 'mean')
    cumstat.add_aggregator('cantitate', 'std')

    # agg = cumstat.group_by('cod_art')

    monthly_stat, weekly_stat = None, None

    cumstat.get_daily_items(item_type='category', type_identifier='branza')

    cumstat.plot_daily_combined()
    cumstat.plot_weekly_combined()

    # plot_all_categories()

    # daily_all = cumstat.cumulate_daily_all_articles()
    # daily_all.plot(x='data', y='valoare')
    # plt.savefig(fr'C:\Users\bas6clj\time-series-forecast\data\combined_valoare.png', dpi=300)

    # cumstat.plot_monthly(74, 'monthly_74.png')
    # cumstat.plot_weekly(74, 'weekly_74.png')
    # cumstat.plot_weekly(1312, 'weekly_1312_.png')
    # cumstat.plot_daily(1312, 'daily_1312_.png')

    # plot_all_articles(True)

    # if monthly_df:
    #     monthly_stat = cumstat.cumulate_monthly_all_articles(fill_missing_dates=False)
    #     monthly_stat.to_csv(monthly_df, index=False)
    #
    # if monthly_stat_out_path:
    #     monthly_stat = cumstat.cumulate_monthly_all_articles() if monthly_stat is None else monthly_stat
    #
    #     monthly_cumstat = CumStat(df=monthly_stat)
    #     monthly_cumstat.add_aggregator('cantitate', 'min')
    #     monthly_cumstat.add_aggregator('cantitate', 'max')
    #     monthly_cumstat.add_aggregator('cantitate', 'mean')
    #     monthly_cumstat.add_aggregator('cantitate', 'std')
    #
    #     monthly_agg = monthly_cumstat.group_by('cod_art')
    #     monthly_agg.to_csv(monthly_stat_out_path, index=False)
    #
    # if weekly_df:
    #     weekly_stat = cumstat.cumulate_weekly_all_articles(fill_missing_dates=False)
    #     weekly_stat.to_csv(weekly_df, index=False)
    #
    # if weekly_stat_out_path:
    #     weekly_stat = cumstat.cumulate_weekly_all_articles() if weekly_stat is None else weekly_stat
    #
    #     weekly_cumstat = CumStat(df=weekly_stat)
    #     weekly_cumstat.add_aggregator('cantitate', 'min')
    #     weekly_cumstat.add_aggregator('cantitate', 'max')
    #     weekly_cumstat.add_aggregator('cantitate', 'mean')
    #     weekly_cumstat.add_aggregator('cantitate', 'std')
    #
    #     weekly_agg = weekly_cumstat.group_by('cod_art').reset_index()
    #     weekly_agg.to_csv(weekly_stat_out_path, index=False)
