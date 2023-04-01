import pandas as pd
import argparse
import datetime
import matplotlib.pyplot as plt


class CumStat:

    """
    A class to quickly apply aggregations on the given dataframe.
    """

    def __init__(self, path_to_csv=None, df=None):
        if path_to_csv and df:
            raise Exception('CumStat object should initialized with either a path to a CSV file or a DataFrame.')
        self.path_to_csv = None
        if path_to_csv:
            self.path_to_csv = path_to_csv
            self.df = pd.read_csv(path_to_csv)
        else:
            self.df = df
        self.monthly_df = None
        self.monthly_df_filled = False
        self.weekly_df = None
        self.weekly_df_filled = False
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

        return self.df.groupby(group_by)\
            .agg(**{f'{agg[1]}_{agg[0]}': (agg[0], agg[1]) for agg in self.aggregators}).reset_index()

    def _create_monthly_df(self):

        """
        Helper function to create the monthly cumulated DataFrame.
        """

        self.df['month'] = self.df['data'].apply(lambda x: x[:7].replace('-', '_'))
        self.monthly_df = self.df.groupby(['cod_art', 'month']).agg(cantitate=('cantitate', 'sum'),
                                                                    pret=('pret', 'mean'),
                                                                    valoare=('valoare', 'sum')).reset_index()

    def _create_weekly_df(self):

        """
        Helper function to create the weekly cumulated DataFrame.
        """

        self.df['week_no'] = self.df['data'].apply(lambda x:
                                                   datetime.date(*[int(s) for s in x.split('-')]).isocalendar())
        self.df['week_no'] = self.df['week_no'].apply(lambda x: '{:d}_{:02d}'.format(x[0], x[1]))
        self.df = self.df.sort_values(by=['cod_art', 'week_no'])

        self.weekly_df = self.df.groupby(['cod_art', 'week_no']).agg(cantitate=('cantitate', 'sum'),
                                                                     pret=('pret', 'mean'),
                                                                     valoare=('valoare', 'sum')).reset_index()

    def cumulate_monthly_all_articles(self, fill_missing_dates=False):

        """
        Function to calculate monthly statistics for all the articles and possibly fill in the missing dates.

        :param fill_missing_dates: whether to fill in the missing dates with 0 sales
        :return: a DataFrame object that contains all the relevant information cumulated for each month
        """

        if self.monthly_df is None:
            self._create_monthly_df()

        if not fill_missing_dates:
            return self.monthly_df.copy()

        cod_arts = self.df['cod_art'].unique()
        start_end_date_df = self.df.groupby('cod_art').agg(start_date=('data', 'min'), end_date=('data', 'max'))

        for cod_art in cod_arts:
            month_no_list = set()
            for i in start_end_date_df.loc[cod_art]:
                month_no = i[:7].replace('-', '_')
                month_no_list.add(month_no)

            date_range = pd.date_range(*start_end_date_df.loc[cod_art].array, freq='M')
            for i in date_range:
                month_no = i.date().isoformat()[:7].replace('-', '_')
                month_no_list.add(month_no)

            cod_art_df = self.monthly_df.query('cod_art == @cod_art').sort_values(by='month')
            for month_no in month_no_list:
                contains = (cod_art_df['month'].eq(month_no)).any()
                if not contains:
                    self.monthly_df.loc[len(self.monthly_df.index)] = [cod_art, month_no, 0, 0, 0]

            self.monthly_df = self.monthly_df.sort_values(by=['cod_art', 'month'])
            self.monthly_df_filled = True

        return self.monthly_df.copy()

    def cumulate_weekly_all_articles(self, fill_missing_dates=False):

        """
        Function to calculate weekly statistics for all the articles and possibly fill in the missing dates.

        :param fill_missing_dates: whether to fill in the missing dates with 0 sales
        :return: a DataFrame object that contains all the relevant information cumulated for each month
        """

        if self.weekly_df is None:
            self._create_weekly_df()

        if not fill_missing_dates:
            return self.weekly_df.copy()

        cod_arts = self.df['cod_art'].unique()
        start_end_date_df = self.df.groupby('cod_art').agg(start_date=('data', 'min'), end_date=('data', 'max'))

        for cod_art in cod_arts:
            week_no_list = set()
            for i in start_end_date_df.loc[cod_art]:
                week_no = datetime.date(*[int(s) for s in i.split('-')]).isocalendar()
                week_no = '{:d}_{:02d}'.format(week_no[0], week_no[1])
                week_no_list.add(week_no)

            date_range = pd.date_range(*start_end_date_df.loc[cod_art].array, freq='W')
            for i in date_range:
                week_no = i.date().isocalendar()
                week_no = '{:d}_{:02d}'.format(week_no[0], week_no[1])
                week_no_list.add(week_no)

            cod_art_df = self.weekly_df.query('cod_art == @cod_art').sort_values(by='week_no')
            for week_no in week_no_list:
                contains = (cod_art_df['week_no'].eq(week_no)).any()
                if not contains:
                    self.weekly_df.loc[len(self.weekly_df.index)] = [cod_art, week_no, 0, 0, 0]

            self.weekly_df = self.weekly_df.sort_values(by=['cod_art', 'week_no'])
            self.weekly_df_filled = True

        return self.weekly_df.copy()

    def cumulate_monthly_article(self, cod_art):

        """
        Calculate monthly statistics with filled missing dates for a single article.
        :param cod_art: No. of the selected article.
        :return: DataFrame with the cumulated monthly statistics.
        """

        if self.monthly_df is None:
            self._create_monthly_df()

        if self.monthly_df_filled:
            return self.monthly_df.query('cod_art == @cod_art')

        cod_art_monthly_df = self.monthly_df.query('cod_art == @cod_art').sort_values(by='month')
        start_end_date_df = self.df.groupby('cod_art').agg(start_date=('data', 'min'), end_date=('data', 'max'))

        month_no_list = set()
        for i in start_end_date_df.loc[cod_art]:
            month_no = i[:7].replace('-', '_')
            month_no_list.add(month_no)

        date_range = pd.date_range(*start_end_date_df.loc[cod_art].array, freq='M')
        for i in date_range:
            month_no = i.date().isoformat()[:7].replace('-', '_')
            month_no_list.add(month_no)

        for month_no in month_no_list:
            contains = (cod_art_monthly_df['month'].eq(month_no)).any()
            if not contains:
                cod_art_monthly_df.loc[len(cod_art_monthly_df.index)] = [cod_art, month_no, 0, 0, 0]

        return cod_art_monthly_df.sort_values(by='month')

    def cumulate_weekly_article(self, cod_art):

        """
        Calculate weekly statistics with filled missing dates for a single article.
        :param cod_art: No. of the selected article.
        :return: DataFrame with the cumulated monthly statistics.
        """

        if self.weekly_df is None:
            self._create_weekly_df()

        if self.weekly_df_filled:
            return self.weekly_df.query('cod_art == @cod_art')

        cod_art_weekly_df = self.weekly_df.query('cod_art == @cod_art').sort_values(by='week_no')
        start_end_date_df = self.df.groupby('cod_art').agg(start_date=('data', 'min'), end_date=('data', 'max'))

        week_no_list = set()
        for i in start_end_date_df.loc[cod_art]:
            week_no = datetime.date(*[int(s) for s in i.split('-')]).isocalendar()
            week_no = '{:d}_{:02d}'.format(week_no[0], week_no[1])
            week_no_list.add(week_no)

        date_range = pd.date_range(*start_end_date_df.loc[cod_art].array, freq='W')
        for i in date_range:
            week_no = i.date().isocalendar()
            week_no = '{:d}_{:02d}'.format(week_no[0], week_no[1])
            week_no_list.add(week_no)

        for week_no in week_no_list:
            contains = (cod_art_weekly_df['week_no'].eq(week_no)).any()
            if not contains:
                cod_art_weekly_df.loc[len(cod_art_weekly_df.index)] = [cod_art, week_no, 0, 0, 0]

        return cod_art_weekly_df.sort_values(by='week_no')

    def plot_monthly(self, cod_art, filename=None):
        monthly_df = self.cumulate_monthly_article(cod_art)
        cod_art_df = monthly_df.query(f'cod_art == {cod_art}')
        cod_art_df.plot(x='month', y='cantitate')

        plt.xticks(range(len(cod_art_df['month'])), cod_art_df['month'], rotation=45)
        plt.ylabel('Cantitate')
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_weekly(self, cod_art, filename=None):
        weekly_df = self.cumulate_weekly_article(cod_art)
        cod_art_df = weekly_df.query(f'cod_art == {cod_art}')
        cod_art_df.plot(x='week_no', y='cantitate')

        plt.xticks(range(len(cod_art_df['week_no'])), cod_art_df['week_no'], rotation=60)
        plt.xticks(fontsize=5)
        plt.ylabel('Cantitate')
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
    agg = cumstat.cumulate_monthly_all_articles()
    # print(agg.to_string())
    agg = cumstat.cumulate_weekly_all_articles()
    # print(agg.to_string())

    weekly_cumstat = CumStat(df=agg)
    weekly_cumstat.add_aggregator('cantitate', 'min')
    weekly_cumstat.add_aggregator('cantitate', 'max')
    weekly_cumstat.add_aggregator('cantitate', 'mean')
    weekly_cumstat.add_aggregator('cantitate', 'std')

    weekly_agg = weekly_cumstat.group_by('cod_art').reset_index()
    print(weekly_agg)

    cumstat.plot_monthly(74, 'monthly_74.png')
    cumstat.plot_weekly(74, 'weekly_74.png')


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

    agg = cumstat.group_by('cod_art')

    monthly_stat, weekly_stat = None, None

    # cumstat.plot_monthly(74, 'monthly_74.png')
    # cumstat.plot_weekly(74, 'weekly_74.png')

    if monthly_df:
        monthly_stat = cumstat.cumulate_monthly_all_articles(fill_missing_dates=False)
        monthly_stat.to_csv(monthly_df, index=False)

    if monthly_stat_out_path:
        monthly_stat = cumstat.cumulate_monthly_all_articles() if monthly_stat is None else monthly_stat

        monthly_cumstat = CumStat(df=monthly_stat)
        monthly_cumstat.add_aggregator('cantitate', 'min')
        monthly_cumstat.add_aggregator('cantitate', 'max')
        monthly_cumstat.add_aggregator('cantitate', 'mean')
        monthly_cumstat.add_aggregator('cantitate', 'std')

        monthly_agg = monthly_cumstat.group_by('cod_art')
        monthly_agg.to_csv(monthly_stat_out_path, index=False)

    if weekly_df:
        weekly_stat = cumstat.cumulate_weekly_all_articles(fill_missing_dates=False)
        weekly_stat.to_csv(weekly_df, index=False)

    if weekly_stat_out_path:
        weekly_stat = cumstat.cumulate_weekly_all_articles() if weekly_stat is None else weekly_stat

        weekly_cumstat = CumStat(df=weekly_stat)
        weekly_cumstat.add_aggregator('cantitate', 'min')
        weekly_cumstat.add_aggregator('cantitate', 'max')
        weekly_cumstat.add_aggregator('cantitate', 'mean')
        weekly_cumstat.add_aggregator('cantitate', 'std')

        weekly_agg = weekly_cumstat.group_by('cod_art').reset_index()
        weekly_agg.to_csv(weekly_stat_out_path, index=False)
