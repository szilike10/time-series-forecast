import statsmodels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from data.data_generator import DataLoader
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import timedelta
from sklearn.metrics import mean_squared_error


def check_stationarity(ts):
    dftest = adfuller(ts)
    adf = dftest[0]
    pvalue = dftest[1]
    critical_value = dftest[4]['5%']

    print('ADF Statistic: %f' % adf)
    print('p-value: %f' % pvalue)
    print('Critical Values:')
    for key, value in dftest[4].items():
        print('\t%s: %.3f' % (key, value))

    if (pvalue < 0.05) and (adf < critical_value):
        print('The series is stationary')
        return True
    else:
        print('The series is NOT stationary')
        return False


def plot_autocorrelation():
    dataloader = DataLoader(path_to_csv=r'C:\Users\bas6clj\time-series-forecast\data\combined.csv')
    column = 'cantitate'

    train, val = dataloader.load_combined_data(column)

    stationary = check_stationarity(train['y'])

    if not stationary:
        fig, (ax1, ax2, ax3) = plt.subplots(3)

        plot_acf(train.y, ax=ax1)
        plot_acf(train.y.diff().dropna(), ax=ax2)
        plot_acf(train.y.diff().diff().dropna(), ax=ax3)
        plt.show()

    plot_pacf(train.y.diff().dropna())
    plt.show()

    plot_acf(train.y.diff().dropna())
    plt.show()


def plot_forecast(y_true, y_pred, column=''):
    fig, ax = plt.subplots()

    ax.plot(y_true['ds'], y_true['y'])
    ax.plot(y_pred['ds'], y_pred['y'], 'r')

    plt.savefig(f'statsmodels_forecast_combined_{column}', dpi=300)

def predict_combined_products():
    dataloader = DataLoader(path_to_csv=r'C:\Users\bas6clj\time-series-forecast\data\combined.csv')
    column = 'cantitate'

    train, val = dataloader.load_combined_data(column,
                                               start_date=pd.to_datetime('2022-01-01'),
                                               end_date=pd.to_datetime('2023-01-01'))

    p, d, q, m = 7, 0, 7, 14

    ARMAmodel = SARIMAX(train['y'], order=(p, d, q), seasonal_order=(p, d, q, m))
    ARMAmodel = ARMAmodel.fit()

    y_pred = ARMAmodel.get_forecast(len(val['ds']))
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df["y"] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df['ds'] = val['ds']
    y_pred_df.index = val.index

    # days_to_forecast = 100
    #
    # y_pred = ARMAmodel.get_forecast(days_to_forecast)
    # y_pred_df = y_pred.conf_int(alpha=0.05)
    # y_pred_df['y'] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    # # print(val['ds'][0])
    # y_pred_df['ds'] = pd.date_range(start=val['ds'].iloc[0], end=val['ds'].iloc[0] + pd.DateOffset(days=days_to_forecast-1), freq='D')


    train.append(val).plot(x='ds', y='y')

    loss = mean_squared_error(val['y'], y_pred_df['y'])
    print('loss = ', loss)

    loss_df = pd.DataFrame.from_dict({'name': [column], 'loss': [loss]})
    loss_df.to_csv(f'statsmodels_combined_{p}_{d}_{q}_{m}_{column}.csv', index=False)


    # print(len(y_pred_df['y']))
    # print(len(val['y']))

    ax = plt.gca()
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    ax.plot(y_pred_df['ds'], y_pred_df['y'], c='red')

    plt.savefig(f'statsmodels_{p}_{d}_{q}_{m}_{column}.png', dpi=300)
    plt.show()

    plot_forecast(val, y_pred_df)



if __name__ == '__main__':
    # plot_autocorrelation()

    predict_combined_products()

