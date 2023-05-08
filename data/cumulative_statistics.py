import pandas as pd
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np


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
        self.monthly_cod_art_df = None
        self.monthly_category_df = None
        self.monthly_df_filled = False
        self.weekly_cod_art_df = None
        self.weekly_category_df = None
        self.weekly_df_filled = False
        self.daily_cod_art_df = None
        self.daily_category_df = None
        self.daily_df_filled = False
        self.aggregators = []

    def get_all_article_ids(self):
        return np.sort(self.df['cod_art'].unique())

    def get_all_categories(self):
        return np.sort(self.df['category'].unique())

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

    def _create_monthly_cod_art_df(self):

        """
        Helper function to create the monthly cumulated DataFrame.
        """

        self.df['month'] = self.df['data'].apply(lambda x: x.strftime('%Y-%m'))
        self.monthly_cod_art_df = self.df.groupby(['cod_art', 'month']).agg(cantitate=('cantitate', 'sum'),
                                                                            pret=('pret', 'mean'),
                                                                            valoare=('valoare', 'sum')).reset_index()
        self.monthly_cod_art_df = self.monthly_cod_art_df.set_index('month')
        self.monthly_cod_art_df.index.name = 'data'

    def _create_monthly_category_df(self):

        """
        Helper function to create the monthly cumulated DataFrame.
        """

        self.df['month'] = self.df['data'].apply(lambda x: x.strftime('%Y-%m'))
        self.monthly_category_df = self.df.groupby(['category', 'month']).agg(cantitate=('cantitate', 'sum'),
                                                                              pret=('pret', 'mean'),
                                                                              valoare=('valoare', 'sum')).reset_index()
        self.monthly_category_df = self.monthly_category_df.set_index('month')
        self.monthly_category_df.index.name = 'data'

    def _create_weekly_cod_art_df(self):

        """
        Helper function to create the weekly cumulated DataFrame.
        """

        self.df['week_no'] = self.df['data'].apply(lambda x: x.strftime('%Y-%U'))
        self.df = self.df.sort_values(by=['cod_art', 'week_no'])

        self.weekly_cod_art_df = self.df.groupby(['cod_art', 'week_no']).agg(cantitate=('cantitate', 'sum'),
                                                                             pret=('pret', 'mean'),
                                                                             valoare=('valoare', 'sum')).reset_index()
        self.weekly_cod_art_df = self.weekly_cod_art_df.set_index('week_no')

    def _create_weekly_category_df(self):

        """
        Helper function to create the weekly cumulated DataFrame.
        """

        self.df['week_no'] = self.df['data'].apply(lambda x: x.strftime('%Y-%U'))
        self.df = self.df.sort_values(by=['category', 'week_no'])

        self.weekly_category_df = self.df.groupby(['category', 'week_no']).agg(cantitate=('cantitate', 'sum'),
                                                                               pret=('pret', 'mean'),
                                                                               valoare=('valoare', 'sum')).reset_index()
        self.weekly_category_df = self.weekly_category_df.set_index('week_no')

    def _create_daily_cod_art_df(self):

        """
        Helper function to create the weekly cumulated DataFrame.
        """

        self.df['day'] = self.df['data']
        self.df = self.df.sort_values(by=['cod_art', 'day'])

        self.daily_cod_art_df = self.df.groupby(['cod_art', 'day']).agg(cantitate=('cantitate', 'sum'),
                                                                        pret=('pret', 'mean'),
                                                                        valoare=('valoare', 'sum'),
                                                                        name=('denumire', 'first')).reset_index()
        self.daily_cod_art_df['data'] = self.daily_cod_art_df['day']
        self.daily_cod_art_df = self.daily_cod_art_df.set_index('day')

    def _create_daily_category_df(self):

        """
        Helper function to create the weekly cumulated DataFrame.
        """

        self.df['day'] = self.df['data']
        self.df = self.df.sort_values(by=['category', 'day'])

        self.daily_category_df = self.df.groupby(['category', 'day']).agg(cantitate=('cantitate', 'sum'),
                                                                          pret=('pret', 'mean'),
                                                                          valoare=('valoare', 'sum'),
                                                                          name=('denumire', 'first')).reset_index()
        self.daily_category_df['data'] = self.daily_category_df['day']
        self.daily_category_df = self.daily_category_df.set_index('day')

    def cumulate_monthly_all_articles(self, fill_missing_dates=False):

        """
        Function to calculate monthly statistics for all the articles and possibly fill in the missing dates.

        :param fill_missing_dates: whether to fill in the missing dates with 0 sales
        :return: a DataFrame object that contains all the relevant information cumulated for each month
        """

        if self.monthly_cod_art_df is None:
            self._create_monthly_cod_art_df()

        if not fill_missing_dates:
            return self.monthly_cod_art_df.copy()

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

            cod_art_df = self.monthly_cod_art_df.query('cod_art == @cod_art').sort_values(by='month')
            for month_no in month_no_list:
                contains = (cod_art_df['month'].eq(month_no)).any()
                if not contains:
                    self.monthly_cod_art_df.loc[len(self.monthly_cod_art_df.index)] = [cod_art, month_no, 0, 0, 0]

            self.monthly_cod_art_df = self.monthly_cod_art_df.sort_values(by=['cod_art', 'month'])
            self.monthly_df_filled = True

        return self.monthly_cod_art_df.copy()

    def cumulate_weekly_all_articles(self, fill_missing_dates=False):

        """
        Function to calculate weekly statistics for all the articles and possibly fill in the missing dates.

        :param fill_missing_dates: whether to fill in the missing dates with 0 sales
        :return: a DataFrame object that contains all the relevant information cumulated for each month
        """

        if self.weekly_cod_art_df is None:
            self._create_weekly_cod_art_df()

        if not fill_missing_dates:
            return self.weekly_cod_art_df.copy()

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

            cod_art_df = self.weekly_cod_art_df.query('cod_art == @cod_art').sort_values(by='week_no')
            for week_no in week_no_list:
                contains = (cod_art_df['week_no'].eq(week_no)).any()
                if not contains:
                    self.weekly_cod_art_df.loc[len(self.weekly_cod_art_df.index)] = [cod_art, week_no, 0, 0, 0]

            self.weekly_cod_art_df = self.weekly_cod_art_df.sort_values(by=['cod_art', 'week_no'])
            self.weekly_df_filled = True

        return self.weekly_cod_art_df.copy()

    # TODO: Adapt cumulative algorithms for date indexes as well
    def cumulate_monthly_article(self, cod_art, fill_missing_data=False):

        """
        Calculate monthly statistics with filled missing dates for a single article.
        :param cod_art: No. of the selected article.
        :param fill_missing_data: whether to fill in the dates that have no sales on them
        :return: DataFrame with the cumulated monthly statistics.
        """

        if self.monthly_cod_art_df is None:
            self._create_monthly_cod_art_df()

        if self.monthly_df_filled:
            return self.monthly_cod_art_df.query('cod_art == @cod_art')

        cod_art_monthly_df = self.monthly_cod_art_df.query('cod_art == @cod_art')
        if not fill_missing_data:
            return cod_art_monthly_df
        start_end_date_df = self.df.groupby('cod_art').agg(start_date=('data', 'min'), end_date=('data', 'max'))

        present_months = {i: True for i in cod_art_monthly_df.index}

        date_range = pd.period_range(*start_end_date_df.loc[cod_art].array, freq='M')

        filled_df = pd.DataFrame.from_dict({'cod_art': [cod_art for _ in range(len(date_range))],
                                            'cantitate': [0 for _ in range(len(date_range))],
                                            'pret': [0 for _ in range(len(date_range))],
                                            'valoare': [0 for _ in range(len(date_range))]})
        filled_df = filled_df.set_index(date_range)

        for month_no in present_months:
            filled_df.loc[month_no] = cod_art_monthly_df.loc[month_no, ['cod_art', 'cantitate', 'pret', 'valoare']]
        filled_df['data'] = date_range.to_timestamp()

        return filled_df

    def cumulate_weekly_article(self, cod_art, fill_missing_data=False):

        """
        Calculate weekly statistics with filled missing dates for a single article.
        :param cod_art: No. of the selected article.
        :param fill_missing_data: whether to fill in the dates that have no sales on them
        :return: DataFrame with the cumulated weekly statistics.
        """

        if self.weekly_cod_art_df is None:
            self._create_weekly_cod_art_df()

        if self.weekly_df_filled:
            return self.weekly_cod_art_df.query('cod_art == @cod_art')

        cod_art_weekly_df = self.weekly_cod_art_df.query('cod_art == @cod_art').sort_values(by='week_no')
        if not fill_missing_data:
            return cod_art_weekly_df
        start_end_date_df = self.df.groupby('cod_art').agg(start_date=('data', 'min'), end_date=('data', 'max'))

        present_weeks = {i: True for i in cod_art_weekly_df.index}

        date_range = pd.period_range(*start_end_date_df.loc[cod_art].array, freq='W')
        date_range = pd.to_datetime([i.start_time for i in date_range])
        date_range_index = date_range.strftime('%Y-%W')

        filled_df = pd.DataFrame.from_dict({'cod_art': [cod_art for _ in range(len(date_range))],
                                            'cantitate': [0 for _ in range(len(date_range))],
                                            'pret': [0 for _ in range(len(date_range))],
                                            'valoare': [0 for _ in range(len(date_range))]})

        filled_df = filled_df.set_index(date_range_index)

        for week_no in present_weeks:
            filled_df.loc[week_no] = cod_art_weekly_df.loc[week_no, ['cod_art', 'cantitate', 'pret', 'valoare']]

        filled_df['data'] = date_range

        return filled_df

    def cumulate_daily_article(self, cod_art, fill_missing_data=False):
        """
        Calculate daily statistics with filled missing dates for a single article.
        :param cod_art: No. of the selected article.
        :param fill_missing_data: whether to fill in the dates that have no sales on them
        :return: DataFrame with the cumulated daily statistics.
        """

        if self.daily_cod_art_df is None:
            self._create_daily_cod_art_df()

        cod_art_daily_df = self.daily_cod_art_df.query('cod_art == @cod_art').sort_values(by='day')
        if not fill_missing_data:
            return cod_art_daily_df
        start_end_date_df = cod_art_daily_df.index[[0, -1]]

        present_days = {i: True for i in cod_art_daily_df.index}

        date_range = pd.period_range(*start_end_date_df, freq='D')
        date_range = pd.to_datetime([i.start_time for i in date_range])
        date_range_index = date_range

        filled_df = pd.DataFrame.from_dict({'cod_art': [cod_art for _ in range(len(date_range))],
                                            'cantitate': [0 for _ in range(len(date_range))],
                                            'pret': [0 for _ in range(len(date_range))],
                                            'valoare': [0 for _ in range(len(date_range))]})

        filled_df = filled_df.set_index(date_range_index)

        for day in present_days:
            filled_df.loc[day] = cod_art_daily_df.loc[day, ['cod_art', 'cantitate', 'pret', 'valoare']]

        filled_df['data'] = date_range

        return filled_df

    def cumulate_daily_category(self, category, fill_missing_data=False):
        """
        Calculate daily statistics with filled missing dates for a single article.
        :param category: No. of the selected article.
        :param fill_missing_data: whether to fill in the dates that have no sales on them
        :return: DataFrame with the cumulated daily statistics.
        """

        if self.daily_category_df is None:
            self._create_daily_category_df()

        category_daily_df = self.daily_category_df.query('category == @category').sort_values(by='day')
        if not fill_missing_data:
            return category_daily_df
        start_end_date_df = category_daily_df.index[[0, -1]]

        present_days = {i: True for i in category_daily_df.index}

        date_range = pd.period_range(*start_end_date_df, freq='D')
        date_range = pd.to_datetime([i.start_time for i in date_range])
        date_range_index = date_range

        filled_df = pd.DataFrame.from_dict({'cod_art': [category for _ in range(len(date_range))],
                                            'cantitate': [0 for _ in range(len(date_range))],
                                            'pret': [0 for _ in range(len(date_range))],
                                            'valoare': [0 for _ in range(len(date_range))]})

        filled_df = filled_df.set_index(date_range_index)

        for day in present_days:
            filled_df.loc[day] = category_daily_df.loc[day, ['category', 'cantitate', 'pret', 'valoare']]

        filled_df['data'] = date_range

        return filled_df

    def plot_article_monthly(self, cod_art, filename=None, fill_missing_data=False):
        monthly_df = self.cumulate_monthly_article(cod_art, fill_missing_data)
        plt.plot(range(len(monthly_df)), monthly_df['cantitate'], label='cantitate')
        plt.legend()

        plt.xticks(range(len(monthly_df)), monthly_df.index, rotation=45)
        plt.ylabel('Cantitate')
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_article_weekly(self, cod_art, filename=None, fill_missing_data=False):
        weekly_df = self.cumulate_weekly_article(cod_art, fill_missing_data)
        weekly_df.plot(y='cantitate', use_index=True)

        plt.xticks(rotation=60)
        plt.xticks(fontsize=10)
        plt.ylabel('Cantitate')
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_article_daily(self, cod_art, filename=None, fill_missing_data=False):
        daily_df = self.cumulate_daily_article(cod_art, fill_missing_data)
        daily_df.plot(y='cantitate', use_index=True)

        plt.xticks(rotation=60)
        plt.xticks(fontsize=10)
        plt.ylabel('Cantitate')
        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

    def plot_category_daily(self, category, filename=None, fill_missing_data=False):
        daily_df = self.cumulate_daily_category(category, fill_missing_data)
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

    cumstat.plot_article_monthly(74, 'monthly_74.png')
    cumstat.plot_article_weekly(74, 'weekly_74.png')


def plot_all_articles(fill_missing_articles=False):
    cumstat = CumStat(path_to_csv=path_to_csv)
    cumstat.add_aggregator('cod_art', 'count')
    agg = cumstat.group_by('cod_art').reset_index()
    for cod_art in agg['cod_art']:
        cumstat.plot_article_daily(cod_art, f'all_articles/filled/daily_{cod_art}.png', fill_missing_articles)


def plot_all_categories(fill_missing_articles=False):
    cumstat = CumStat(path_to_csv=path_to_csv)
    categories = cumstat.get_all_categories()
    for category in categories:
        cumstat.plot_category_daily(category, f'all_categories/not_filled/daily_{category}.png', fill_missing_articles)


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

    plot_all_categories()

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
