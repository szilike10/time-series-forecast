import pandas as pd
import argparse
import datetime


class CumStat:
    """
    A class to quickly apply aggregations on the given dataframe.
    """

    def __init__(self, path_to_csv=None, df=None):
        if path_to_csv and df or (path_to_csv is None and df is None):
            raise Exception('CumStat object should initialized with either a path to a CSV file or a DataFrame.')
        self.path_to_csv = None
        if path_to_csv:
            self.path_to_csv = path_to_csv
            self.df = pd.read_csv(path_to_csv)
        else:
            self.df = df
        self.aggregators = []

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
        return self.df.groupby(group_by).agg(**{f'{agg[1]}_{agg[0]}': (agg[0], agg[1]) for agg in self.aggregators})

    # TODO: 1. Add functionality to cumulate these aggregations on weekends or on end of the month.

    def cumulate_monthly(self):
        """
        Function to calculate monthly statistics.

        :return: a DataFrame object that contains all the relevant information cumulated for each month
        """
        self.df['month'] = self.df['data'].apply(lambda x: x[:7])
        return self.df.groupby(['cod_art', 'month']).agg(cantitate=('cantitate', 'sum'),
                                                         pret=('pret', 'mean'),
                                                         valoare=('valoare', 'sum'))

    def cumulate_weekly(self):
        """
        Function to calculate weekly statistics.

       :return: a DataFrame object that contains all the relevant information cumulated for each month
       """
        self.df['week_no'] = self.df['data'].apply(lambda x:
                                                   datetime.date(*[int(s) for s in x.split('-')]).isocalendar()[1])
        return self.df.groupby(['cod_art', 'week_no']).agg(cantitate=('cantitate', 'sum'),
                                                           pret=('pret', 'mean'),
                                                           valoare=('valoare', 'sum'))


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
    agg = cumstat.cumulate_monthly()
    # print(agg.to_string())
    agg = cumstat.cumulate_weekly()
    # print(agg.to_string())

    weekly_cumstat = CumStat(df=agg)
    weekly_cumstat.add_aggregator('cantitate', 'min')
    weekly_cumstat.add_aggregator('cantitate', 'max')
    weekly_cumstat.add_aggregator('cantitate', 'mean')
    weekly_cumstat.add_aggregator('cantitate', 'std')

    weekly_agg = weekly_cumstat.group_by('cod_art').reset_index()
    print(weekly_agg)


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

    agg = cumstat.group_by('cod_art').reset_index()

    monthly_stat, weekly_stat = None, None

    if monthly_df:
        monthly_stat = cumstat.cumulate_monthly().reset_index()
        monthly_stat.to_csv(monthly_df, index=False)

    if monthly_stat_out_path:
        monthly_stat = cumstat.cumulate_monthly() if monthly_stat is not None else monthly_stat

        monthly_cumstat = CumStat(df=monthly_stat)
        monthly_cumstat.add_aggregator('cantitate', 'min')
        monthly_cumstat.add_aggregator('cantitate', 'max')
        monthly_cumstat.add_aggregator('cantitate', 'mean')
        monthly_cumstat.add_aggregator('cantitate', 'std')

        monthly_agg = monthly_cumstat.group_by('cod_art').reset_index()
        monthly_agg.to_csv(monthly_stat_out_path, index=False)

    if weekly_df:
        weekly_stat = cumstat.cumulate_weekly().reset_index()
        weekly_stat.to_csv(weekly_df, index=False)

    if weekly_stat_out_path:
        weekly_stat = cumstat.cumulate_weekly() if weekly_stat is not None else weekly_stat

        weekly_cumstat = CumStat(df=weekly_stat)
        weekly_cumstat.add_aggregator('cantitate', 'min')
        weekly_cumstat.add_aggregator('cantitate', 'max')
        weekly_cumstat.add_aggregator('cantitate', 'mean')
        weekly_cumstat.add_aggregator('cantitate', 'std')

        weekly_agg = weekly_cumstat.group_by('cod_art').reset_index()
        weekly_agg.to_csv(weekly_stat_out_path, index=False)
