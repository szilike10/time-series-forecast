from copy import deepcopy

import numpy as np
import pandas as pd


def fill_missing_dates(df, date_col, target_cols, frequency='D'):
    if type(target_cols) is not list:
        target_cols = [target_cols]

    df = deepcopy(df)
    df['data'] = pd.to_datetime(df[date_col])

    min_date = df['data'].min()
    max_date = df['data'].max()

    start_date = min_date.to_period('W-MON').start_time
    end_date = max_date.to_period('W-SUN').end_time

    if min_date == max_date:
        return df

    date_range = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq=frequency))
    length = len(date_range)

    filled = pd.DataFrame({col: [df[col].values[0] for i in range(length)] for col in df.columns})
    filled = filled.set_index(date_range)
    filled['data'] = date_range

    for target in target_cols:
        filled[target] = np.zeros(length)
        for date, value in zip(df[date_col], df[target]):
            filled.loc[date, target] = value

    # this condition might be unnecessary, if so delete start_date and end_date
    if frequency == 'D':
        filled = filled.query('data >= @min_date')
        filled = filled.query('data <= @max_date')

    return filled


if __name__ == '__main__':
    df = pd.read_csv(fr'C:\Users\bas6clj\time-series-forecast\data\combined.csv')

    fill_missing_dates(df[df['cod_art'] == 675], 'data', 'valoare')
