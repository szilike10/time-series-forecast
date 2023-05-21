import math
import statsmodels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from data.data_generator import DataLoader
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from utils.path_handling import handle_parent_path
from tqdm import tqdm


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


def plot_autocorrelation(frequency, column='valoare'):
    dataloader = DataLoader(path_to_csv=r'../../data/combined.csv')

    # train, val = dataloader.load_combined_data(column)

    train, val = dataloader.load_data(frequency=frequency,
                                      value_type=column,
                                      start_date=pd.to_datetime('2022-01-01'),
                                      end_date=pd.to_datetime('2023-01-01'))
    data = pd.concat([train, val], axis=0)

    stationary = check_stationarity(data['y'])

    if not stationary:
        fig, (ax1, ax2, ax3) = plt.subplots(3)

        plot_acf(data['y'], ax=ax1)
        plot_acf(data['y'].diff().dropna(), ax=ax2)
        plot_acf(data['y'].diff().diff().dropna(), ax=ax3)
        plt.show()

    plot_pacf(data['y'].diff().dropna())
    plt.show()

    plot_acf(data['y'].diff().dropna())
    plt.show()


def plot_forecast(y_true, y_pred, frequency, column, p=None, d=None, q=None, m=None, draw_plot=False):
    fig, ax = plt.subplots()

    ax.plot(y_true['ds'], y_true['y'])
    ax.plot(y_pred['ds'], y_pred['y'], 'r')

    ax.plot(y_pred['ds'], y_pred['lower y'], c='red', alpha=0.5, linewidth=1)
    ax.plot(y_pred['ds'], y_pred['upper y'], c='red', alpha=0.5, linewidth=1)
    ax.fill_between(y_pred['ds'], y_pred['lower y'], y_pred['upper y'], color='red', alpha=0.1)

    val_plot_filename = f'charts/{frequency}/{column}/statsmodels_combined_{p}_{d}_{q}_{m}_val.png'
    handle_parent_path(val_plot_filename)
    plt.savefig(val_plot_filename, dpi=300)

    if draw_plot:
        plt.show()


def predict_combined_products(frequency='daily', column='valoare'):
    dataloader = DataLoader(path_to_csv=r'../../data/combined.csv')

    train, val = dataloader.load_data(frequency=frequency,
                                      value_type=column,
                                      start_date=pd.to_datetime('2022-01-01'),
                                      end_date=pd.to_datetime('2023-01-01'))

    p, d, q, m = (7, 0, 7, 14) if frequency == 'daily' else (2, 1, 1, 8)

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

    data = pd.concat([train, val], axis=0)

    fig, ax = plt.subplots()
    ax.plot(data['ds'], data['y'])

    loss = mean_squared_error(val['y'], y_pred_df['y'])
    print('loss = ', loss)

    loss_df = pd.DataFrame.from_dict({'name': [column], 'loss': [loss], 'adjusted_loss': [loss * 0.000001 * ARMAmodel.bic * ARMAmodel.aic]})
    loss_df_filename = f'charts/{frequency}/{column}/statsmodels_combined_{p}_{d}_{q}_{m}.csv'
    handle_parent_path(loss_df_filename)
    loss_df.to_csv(loss_df_filename, index=False)

    fig.set_size_inches(8, 5)
    ax.plot(y_pred_df['ds'], y_pred_df['y'], c='red')

    # plot uncertainty
    ax.plot(y_pred_df['ds'], y_pred_df['lower y'], c='red', alpha=0.5, linewidth=1)
    ax.plot(y_pred_df['ds'], y_pred_df['upper y'], c='red', alpha=0.5, linewidth=1)
    ax.fill_between(y_pred_df['ds'], y_pred_df['lower y'], y_pred_df['upper y'], color='red', alpha=0.1)

    train_val_plot_filename = f'charts/{frequency}/{column}/statsmodels_{p}_{d}_{q}_{m}.png'
    handle_parent_path(train_val_plot_filename)
    plt.savefig(train_val_plot_filename, dpi=300)
    plt.show()

    plot_forecast(val, y_pred_df, frequency=frequency, p=p, d=d, q=q, m=m, column=column)


def grid_search_parameters(frequency='daily', column='valoare'):
    dataloader = DataLoader(path_to_csv=r'../../data/combined.csv')

    train, val = dataloader.load_data(frequency=frequency,
                                      value_type=column,
                                      start_date=pd.to_datetime('2022-01-01'),
                                      end_date=pd.to_datetime('2023-01-01'))

    data = pd.concat([train, val], axis=0)

    max_p, max_d, max_q, s_params = 10, 0, 10, [2, 4, 8, 12] if frequency == 'weekly' else [7, 14, 21, 28]

    best_non_seasonal = {
        'p': 1,
        'd': 0,
        'q': 1,
        'bic': math.inf,
        'aic': math.inf,
        'loss': math.inf,
        'adjusted_loss': math.inf
    }

    stationary = check_stationarity(data['y'])
    if not stationary:
        max_d = 2

    non_seasonal_df = pd.DataFrame(columns=['p', 'd', 'q', 'loss', 'bic', 'aic', 'adjusted_loss'])
    non_seasonal_df_filename = f'charts/grid_search/{frequency}/{column}/non_seasonal/non_seasonal_search_results.csv'
    df_index = 0

    best_loss = math.inf

    for p in tqdm(range(1, max_p + 1), desc='p'):
        for q in tqdm(range(1, max_q + 1), desc='q'):
            for d in tqdm(range(0, max_d + 1), desc='d'):

                print('Testing for: ', p, d, q)

                try:
                    ARMAmodel = SARIMAX(train['y'], order=(p, d, q))
                    ARMAmodel = ARMAmodel.fit(disp=0)
                except:
                    continue

                y_pred = ARMAmodel.get_forecast(len(val['ds']))
                y_pred_df = y_pred.conf_int(alpha=0.05)
                y_pred_df["y"] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
                y_pred_df['ds'] = val['ds']
                y_pred_df.index = val.index

                loss = mean_squared_error(val['y'], y_pred_df['y'])
                adjusted_loss = ARMAmodel.bic * ARMAmodel.aic * loss

                non_seasonal_df.loc[df_index] = [p, d, q, loss, ARMAmodel.bic, ARMAmodel.aic, adjusted_loss]
                df_index += 1
                handle_parent_path(non_seasonal_df_filename)
                non_seasonal_df.to_csv(non_seasonal_df_filename)

                if loss < best_loss:
                    best_loss = loss
                    best_non_seasonal['p'] = p
                    best_non_seasonal['d'] = d
                    best_non_seasonal['q'] = q
                    best_non_seasonal['loss'] = loss
                    best_non_seasonal['adjusted_loss'] = adjusted_loss

                if ARMAmodel.bic < best_non_seasonal['bic']:
                    best_non_seasonal['bic'] = ARMAmodel.bic

                if ARMAmodel.aic < best_non_seasonal['aic']:
                    best_non_seasonal['aic'] = ARMAmodel.aic

                fig, ax = plt.subplots()
                fig.set_size_inches(8, 5)

                ax.plot(val['ds'], val['y'])
                ax.plot(y_pred_df['ds'], y_pred_df['y'], c='red')

                # plot uncertainty
                ax.plot(y_pred_df['ds'], y_pred_df['lower y'], c='red', alpha=0.5, linewidth=1)
                ax.plot(y_pred_df['ds'], y_pred_df['upper y'], c='red', alpha=0.5, linewidth=1)
                ax.fill_between(y_pred_df['ds'], y_pred_df['lower y'], y_pred_df['upper y'], color='red', alpha=0.1)

                train_val_plot_filename = f'charts/grid_search/{frequency}/{column}/non_seasonal/statsmodels_p={p}_d={d}_q={q}_loss={loss:.2f}.png'
                handle_parent_path(train_val_plot_filename)
                plt.savefig(train_val_plot_filename, dpi=300)

    print('Best non-seasonal values: ', best_non_seasonal)

    non_seasonal_df.to_csv(non_seasonal_df_filename)

    best_seasonal = {
        'p': 1,
        'd': 0,
        'q': 1,
        's': 2,
        'bic': math.inf,
        'aic': math.inf,
        'loss': math.inf,
        'adjusted_loss': math.inf
    }

    seasonal_df = pd.DataFrame(columns=['p', 'd', 'q', 's', 'loss', 'bic', 'aic', 'adjusted_loss'])
    seasonal_df_filename = f'charts/grid_search/{frequency}/{column}/seasonal/seasonal_search_results.csv'
    df_index = 0

    best_loss = math.inf

    for s in tqdm(s_params, desc='s'):
        for p in tqdm(range(1, max_p + 1), desc='p'):
            for q in tqdm(range(1, max_q + 1), desc='q'):
                for d in tqdm(range(0, max_d + 1), desc='d'):

                    print('Testing for: ', p, d, q, s)

                    try:
                        ARMAmodel = SARIMAX(train['y'],
                                            order=(best_non_seasonal['p'], best_non_seasonal['d'], best_non_seasonal['q']),
                                            # order=(10, 0, 10),
                                            seasonal_order=(p, d, q, s))
                        ARMAmodel = ARMAmodel.fit(disp=0)
                    except ValueError:
                        continue

                    y_pred = ARMAmodel.get_forecast(len(val['ds']))
                    y_pred_df = y_pred.conf_int(alpha=0.05)
                    y_pred_df["y"] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
                    y_pred_df['ds'] = val['ds']
                    y_pred_df.index = val.index

                    loss = mean_squared_error(val['y'], y_pred_df['y'])
                    adjusted_loss = ARMAmodel.bic * ARMAmodel.aic * loss

                    seasonal_df.loc[df_index] = [p, d, q, s, loss, ARMAmodel.bic, ARMAmodel.aic, adjusted_loss]
                    df_index += 1
                    handle_parent_path(seasonal_df_filename)
                    seasonal_df.to_csv(seasonal_df_filename)

                    if loss < best_loss:
                        best_loss = loss
                        best_non_seasonal['p'] = p
                        best_non_seasonal['d'] = d
                        best_non_seasonal['q'] = q
                        best_non_seasonal['s'] = s
                        best_non_seasonal['loss'] = loss
                        best_non_seasonal['adjusted_loss'] = adjusted_loss

                    if ARMAmodel.bic < best_non_seasonal['bic']:
                        best_non_seasonal['bic'] = ARMAmodel.bic

                    if ARMAmodel.aic < best_non_seasonal['aic']:
                        best_non_seasonal['aic'] = ARMAmodel.aic

                    fig, ax = plt.subplots()
                    fig.set_size_inches(8, 5)

                    ax.plot(val['ds'], val['y'])
                    ax.plot(y_pred_df['ds'], y_pred_df['y'], c='red')

                    # plot uncertainty
                    ax.plot(y_pred_df['ds'], y_pred_df['lower y'], c='red', alpha=0.5, linewidth=1)
                    ax.plot(y_pred_df['ds'], y_pred_df['upper y'], c='red', alpha=0.5, linewidth=1)
                    ax.fill_between(y_pred_df['ds'], y_pred_df['lower y'], y_pred_df['upper y'], color='red', alpha=0.1)

                    train_val_plot_filename = f'charts/grid_search/{frequency}/{column}/seasonal/statsmodels_p={p}_d={d}_q={q}_s={s}_loss={loss:.2f}.png'
                    handle_parent_path(train_val_plot_filename)
                    plt.savefig(train_val_plot_filename, dpi=300)

    print('Best seasonal values: ', best_non_seasonal)
    seasonal_df.to_csv(seasonal_df_filename)


if __name__ == '__main__':
    frequency = 'weekly'
    column = 'valoare'

    # plot_autocorrelation(frequency=frequency, column=column)
    # predict_combined_products(frequency=frequency, column=column)

    grid_search_parameters(frequency=frequency, column=column)
