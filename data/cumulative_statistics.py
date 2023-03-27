import pandas as pd
import argparse


class CumStat:
    """
    A class to quickly apply aggregations on the given dataframe.
    """

    def __init__(self, path_to_csv):
        self.path_to_csv = path_to_csv
        self.df = pd.read_csv(path_to_csv)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv', type=str)
    args = parser.parse_args()

    path_to_csv = args.path_to_csv

    cumstat = CumStat(path_to_csv=path_to_csv)
    cumstat.add_aggregator('data', 'min')
    cumstat.add_aggregator('data', 'max')
    cumstat.add_aggregator('cantitate', 'min')
    cumstat.add_aggregator('cantitate', 'max')
    cumstat.add_aggregator('cantitate', 'mean')
    cumstat.add_aggregator('cantitate', 'std')

    agg = cumstat.group_by('cod_art')
    print(agg.to_string())
