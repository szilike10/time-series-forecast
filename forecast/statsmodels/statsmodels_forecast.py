import math
import statsmodels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_model import ARMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from data.data_generator import DataLoader
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from utils.path_handling import handle_parent_path
from tqdm import tqdm

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', UserWarning)


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


def plot_forecast(y_true, y_pred, filename, draw_plot=False):
    fig, ax = plt.subplots()

    ax.plot(y_true['ds'], y_true['y'], linewidth=1)
    ax.plot(y_pred['ds'], y_pred['y'], 'r', linewidth=1)

    ax.plot(y_pred['ds'], y_pred['lower y'], c='red', alpha=0.5, linewidth=0.5)
    ax.plot(y_pred['ds'], y_pred['upper y'], c='red', alpha=0.5, linewidth=0.5)
    ax.fill_between(y_pred['ds'], y_pred['lower y'], y_pred['upper y'], color='red', alpha=0.1)

    handle_parent_path(filename)
    plt.savefig(filename, dpi=300)

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

    loss_df = pd.DataFrame.from_dict(
        {'name': [column], 'loss': [loss], 'adjusted_loss': [loss * 0.000001 * ARMAmodel.bic * ARMAmodel.aic]})
    loss_df_filename = f'charts/{frequency}/{column}/statsmodels_combined_{p}_{d}_{q}_{m}.csv'
    handle_parent_path(loss_df_filename)
    loss_df.to_csv(loss_df_filename, index=False)

    train_val_plot_filename = f'charts/{frequency}/{column}/statsmodels_{p}_{d}_{q}_{m}.png'
    plot_forecast(pd.concat([train, val]), y_pred_df, train_val_plot_filename)

    val_plot_filename = f'charts/{frequency}/{column}/statsmodels_combined_{p}_{d}_{q}_{m}_val.png'
    plot_forecast(val, y_pred_df, val_plot_filename)


def grid_search(train, val, type_identifier, frequency='daily', column='valoare'):
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
        max_d = 1

    non_seasonal_df = pd.DataFrame(columns=['p', 'd', 'q', 'loss', 'bic', 'aic'])
    non_seasonal_df_filename = f'charts/grid_search/{frequency}/{column}/{type_identifier}/non_seasonal/non_seasonal_search_results_{type_identifier}.csv'
    df_index = 0

    best_loss = math.inf

    for p in tqdm(range(1, max_p + 1), desc='p'):
        for q in tqdm(range(1, max_q + 1), desc='q'):
            for d in tqdm(range(0, max_d + 1), desc='d'):
                try:
                    ARMAmodel = SARIMAX(train['y'], order=(p, d, q))
                    ARMAmodel = ARMAmodel.fit(disp=0)
                except:
                    continue

                y_pred = ARMAmodel.get_forecast(len(val['ds']))
                y_pred_df = y_pred.conf_int(alpha=0.05)
                y_pred_df['y'] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
                y_pred_df['ds'] = val['ds']
                y_pred_df.index = val.index

                loss = mean_squared_error(val['y'], y_pred_df['y'])
                # adjusted_loss = ARMAmodel.bic * ARMAmodel.aic * loss

                non_seasonal_df.loc[df_index] = [p, d, q, loss, ARMAmodel.bic, ARMAmodel.aic]
                df_index += 1
                handle_parent_path(non_seasonal_df_filename)
                non_seasonal_df.to_csv(non_seasonal_df_filename)

                if loss < best_loss:
                    best_loss = loss
                    best_non_seasonal['p'] = p
                    best_non_seasonal['d'] = d
                    best_non_seasonal['q'] = q
                    best_non_seasonal['loss'] = loss
                    # best_non_seasonal['adjusted_loss'] = adjusted_loss

                if ARMAmodel.bic < best_non_seasonal['bic']:
                    best_non_seasonal['bic'] = ARMAmodel.bic

                if ARMAmodel.aic < best_non_seasonal['aic']:
                    best_non_seasonal['aic'] = ARMAmodel.aic

                val_plot_filename = f'charts/grid_search/{frequency}/{column}/{type_identifier}/non_seasonal/statsmodels_p={p}_d={d}_q={q}_loss={loss:.4f}.png'
                plot_forecast(val, y_pred_df, val_plot_filename)

    print('Best non-seasonal values: ', best_non_seasonal)
    ARMAmodel = SARIMAX(train['y'], order=(best_non_seasonal['p'], best_non_seasonal['d'], best_non_seasonal['q']))
    ARMAmodel = ARMAmodel.fit(disp=0)

    y_pred = ARMAmodel.get_forecast(len(val['ds']))
    y_pred_df = y_pred.conf_int(alpha=0.05)
    y_pred_df['y'] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    y_pred_df['ds'] = val['ds']
    loss = mean_squared_error(val['y'], y_pred_df['y'])

    p, d, q, loss = best_non_seasonal['p'], best_non_seasonal['d'], best_non_seasonal['q'], best_non_seasonal['loss']

    train_val_plot_filename = f'charts/grid_search/{frequency}/{column}/{type_identifier}/non_seasonal/best_statsmodels_p={p}_d={d}_q={q}_loss={loss:.4f}.png'
    plot_forecast(pd.concat([train, val]), y_pred_df, train_val_plot_filename)

    non_seasonal_df.to_csv(non_seasonal_df_filename)

    # best_seasonal = {
    #     'p': 1,
    #     'd': 0,
    #     'q': 1,
    #     's': 2,
    #     'bic': math.inf,
    #     'aic': math.inf,
    #     'loss': math.inf,
    #     'adjusted_loss': math.inf
    # }
    #
    # seasonal_df = pd.DataFrame(columns=['p', 'd', 'q', 's', 'loss', 'bic', 'aic', 'adjusted_loss'])
    # seasonal_df_filename = f'charts/grid_search/{frequency}/{column}/seasonal/seasonal_search_results.csv'
    # df_index = 0
    #
    # best_loss = math.inf
    #
    # for s in tqdm(s_params, desc='s'):
    #     for p in tqdm(range(1, max_p + 1), desc='p'):
    #         for q in tqdm(range(1, max_q + 1), desc='q'):
    #             for d in tqdm(range(0, max_d + 1), desc='d'):
    #
    #                 print('Testing for: ', p, d, q, s)
    #
    #                 try:
    #                     ARMAmodel = SARIMAX(train['y'],
    #                                         order=(best_non_seasonal['p'], best_non_seasonal['d'], best_non_seasonal['q']),
    #                                         # order=(10, 0, 10),
    #                                         seasonal_order=(p, d, q, s))
    #                     ARMAmodel = ARMAmodel.fit(disp=0)
    #                 except ValueError:
    #                     continue
    #
    #                 y_pred = ARMAmodel.get_forecast(len(val['ds']))
    #                 y_pred_df = y_pred.conf_int(alpha=0.05)
    #                 y_pred_df["y"] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
    #                 y_pred_df['ds'] = val['ds']
    #                 y_pred_df.index = val.index
    #
    #                 loss = mean_squared_error(val['y'], y_pred_df['y'])
    #                 adjusted_loss = ARMAmodel.bic * ARMAmodel.aic * loss
    #
    #                 seasonal_df.loc[df_index] = [p, d, q, s, loss, ARMAmodel.bic, ARMAmodel.aic, adjusted_loss]
    #                 df_index += 1
    #                 handle_parent_path(seasonal_df_filename)
    #                 seasonal_df.to_csv(seasonal_df_filename)
    #
    #                 if loss < best_loss:
    #                     best_loss = loss
    #                     best_non_seasonal['p'] = p
    #                     best_non_seasonal['d'] = d
    #                     best_non_seasonal['q'] = q
    #                     best_non_seasonal['s'] = s
    #                     best_non_seasonal['loss'] = loss
    #                     best_non_seasonal['adjusted_loss'] = adjusted_loss
    #
    #                 if ARMAmodel.bic < best_non_seasonal['bic']:
    #                     best_non_seasonal['bic'] = ARMAmodel.bic
    #
    #                 if ARMAmodel.aic < best_non_seasonal['aic']:
    #                     best_non_seasonal['aic'] = ARMAmodel.aic
    #
    #                 fig, ax = plt.subplots()
    #                 fig.set_size_inches(8, 5)
    #
    #                 ax.plot(val['ds'], val['y'])
    #                 ax.plot(y_pred_df['ds'], y_pred_df['y'], c='red')
    #
    #                 # plot uncertainty
    #                 ax.plot(y_pred_df['ds'], y_pred_df['lower y'], c='red', alpha=0.5, linewidth=1)
    #                 ax.plot(y_pred_df['ds'], y_pred_df['upper y'], c='red', alpha=0.5, linewidth=1)
    #                 ax.fill_between(y_pred_df['ds'], y_pred_df['lower y'], y_pred_df['upper y'], color='red', alpha=0.1)
    #
    #                 train_val_plot_filename = f'charts/grid_search/{frequency}/{column}/seasonal/statsmodels_p={p}_d={d}_q={q}_s={s}_loss={loss:.2f}.png'
    #                 handle_parent_path(train_val_plot_filename)
    #                 plt.savefig(train_val_plot_filename, dpi=300)
    #
    # print('Best seasonal values: ', best_non_seasonal)
    # seasonal_df.to_csv(seasonal_df_filename)

    return loss, p, d, q


def grid_search_item_types(item_type=None, frequency='daily', column='valoare'):
    dataloader = DataLoader(path_to_csv=r'../../data/combined.csv')

    mse_loss = []
    p_values = []
    d_values = []
    q_values = []

    relevant_type_identifiers = []

    type_identifiers = [None]
    if item_type == 'category':
        type_identifiers = dataloader.get_categories()
    elif item_type == 'cod_art':
        type_identifiers = dataloader.get_article_ids()

    for type_identifier in type_identifiers:
        print(type_identifier, ':')

        train, val = dataloader.load_data(frequency=frequency,
                                          value_type=column,
                                          item_type=item_type,
                                          type_identifier=type_identifier,
                                          start_date=pd.to_datetime('2022-01-01'),
                                          end_date=pd.to_datetime('2023-01-01'),
                                          min_length=20)

        if len(train) + len(val) > 0:
            relevant_type_identifiers.append(type_identifier)

            print(type_identifier)
            loss, p, d, q = grid_search(train, val, type_identifier, frequency, column)
            mse_loss.append(loss)
            p_values.append(p)
            d_values.append(d)
            q_values.append(q)

    df = pd.DataFrame.from_dict({
        str(item_type): relevant_type_identifiers,
        'mse_loss': mse_loss,
        'p': p_values,
        'd': d_values,
        'q': q_values})
    filename = f'charts/grid_search/{frequency}/{column}/statsmodels_{item_type}_grid_search.csv'
    handle_parent_path(filename)
    df.to_csv(filename, index=True)


def compare_to_mean(item_type=None, frequency='daily', column='valoare'):
    dataloader = DataLoader(path_to_csv=r'../../data/combined.csv')

    mse_loss = []
    statsmodels_losses = []
    diffs = []
    percentages = []
    improved = []
    relevant_type_identifiers = []
    ps, ds, qs = [], [], []

    type_identifiers = [None]
    if item_type == 'category':
        type_identifiers = dataloader.get_categories()
    elif item_type == 'cod_art':
        type_identifiers = dataloader.get_article_ids()

    df = pd.read_csv(f'charts/grid_search/{frequency}/{column}/statsmodels_{item_type}_grid_search.csv', index_col=str(item_type))

    for type_identifier in type_identifiers:

        train, val = dataloader.load_data(frequency=frequency,
                                          value_type=column,
                                          item_type=item_type,
                                          type_identifier=type_identifier,
                                          start_date=pd.to_datetime('2022-01-01'),
                                          end_date=pd.to_datetime('2023-01-01'))

        if len(train) + len(val) > 100:
            relevant_type_identifiers.append(type_identifier)

            mean = train['y'].mean()
            mean_val = pd.DataFrame.from_dict({'ds': val['ds'], 'y': mean * np.ones(len(val))})

            loss = mean_squared_error(val['y'], mean_val['y'])
            statsmodels_loss = df.loc[type_identifier, 'mse_loss']
            diff = statsmodels_loss - loss

            ps.append(df.loc[type_identifier, 'p'])
            ds.append(df.loc[type_identifier, 'd'])
            qs.append(df.loc[type_identifier, 'q'])

            improved.append(1 if diff < 0 else -1)
            percentages.append(-diff / loss)

            mse_loss.append(loss)
            statsmodels_losses.append(statsmodels_loss)
            diffs.append(diff)

    df = pd.DataFrame.from_dict({
        str(item_type): relevant_type_identifiers,
        'mse_loss_of_mean': mse_loss,
        'statsmodels_loss': statsmodels_losses,
        'diff': diffs,
        'p': ps,
        'd': ds,
        'q': qs,
        'improved': improved,
        'improved %': percentages
    })
    filename = f'charts/comparison_to_mean/{frequency}/{column}/statsmodels_{item_type}_compare_to_mean.csv'
    handle_parent_path(filename)
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    frequency = 'daily'
    column = 'valoare'

    plot_autocorrelation(frequency=frequency, column=column)
    predict_combined_products(frequency=frequency, column=column)

    # grid_search_item_types(item_type='category', frequency=frequency, column=column)

    # compare_to_mean(item_type='category', frequency=frequency, column=column)
