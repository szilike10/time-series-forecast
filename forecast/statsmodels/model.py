import math

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

from data.data_generator import DataLoader
from forecast.model import ForecastingModel
from forecast.statsmodels.statsmodels_config import StatsmodelsConfig


class StatModel(ForecastingModel):
    def __init__(self, cfg: StatsmodelsConfig):
        super().__init__()

        self.cfg = cfg

        self.dataloader = DataLoader(self.cfg.combined_csv_path)

        self.type_identifier = self.cfg.type_identifier

        self.train, self.val = self.dataloader.load_data(frequency=self.cfg.frequency,
                                               value_type=self.cfg.target,
                                               item_type='_'.join(self.cfg.group_identifiers),
                                               type_identifier=self.cfg.type_identifier,
                                               start_date=self.cfg.start_date,
                                               end_date=self.cfg.end_date)

    def check_stationarity(self):

        ts = pd.concat([self.train, self.val], axis=0)

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

    def plot_autocorrelations(self):
        data = pd.concat([self.train, self.val], axis=0)

        fig_d, (ax1, ax2, ax3) = plt.subplots(3)

        plot_acf(data['y'], ax=ax1)
        plot_acf(data['y'].diff().dropna(), ax=ax2)
        plot_acf(data['y'].diff().diff().dropna(), ax=ax3)

        fig_p = plot_pacf(data['y'].diff().dropna())

        fig_q = plot_acf(data['y'].diff().dropna())

        plt.show()

    def fit(self, type_identifier=None, p=None, d=None, q=None, P=None, D=None, Q=None, S=None):
        p, d, q, S = (7, 0, 7, 14) if self.cfg.frequency == 'daily' else (2, 1, 1, 8)

        model = SARIMAX(self.train['y'], order=(p, d, q), seasonal_order=(P, D, Q, S))
        model = model.fit()

        return model

    def predict(self, model, val):
        y_pred = model.get_forecast(len(self.val['ds']))
        y_pred_df = y_pred.conf_int(alpha=0.05)
        y_pred_df['y'] = model.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
        y_pred_df['ds'] = val['ds']
        y_pred_df.index = val.index

        return y_pred_df

    def grid_search(self, max_p, max_q):
        best_non_seasonal = {
            'p': 1,
            'd': 0,
            'q': 1,
            'bic': math.inf,
            'aic': math.inf,
            'loss': math.inf,
            'adjusted_loss': math.inf
        }

        max_d = 0
        stationary = self.check_stationarity()
        if not stationary:
            max_d = 1

        search_df = pd.DataFrame(columns=['p', 'd', 'q', 'loss', 'bic', 'aic'])
        df_index = 0

        best_loss = math.inf

        for p in tqdm(range(1, max_p + 1), desc='p'):
            for q in tqdm(range(1, max_q + 1), desc='q'):
                for d in tqdm(range(0, max_d + 1), desc='d'):
                    try:
                        model = SARIMAX(self.train['y'], order=(p, d, q))
                        model = model.fit(disp=0)
                    except:
                        continue

                    y_pred_df = self.predict(model, self.val)

                    loss = mean_squared_error(self.val['y'], y_pred_df['y'])
                    # adjusted_loss = ARMAmodel.bic * ARMAmodel.aic * loss

                    search_df.loc[df_index] = [p, d, q, loss, model.bic, model.aic]
                    df_index += 1

                    if loss < best_loss:
                        best_loss = loss
                        search_df['p'] = p
                        search_df['d'] = d
                        search_df['q'] = q
                        search_df['loss'] = loss
                        # best_non_seasonal['adjusted_loss'] = adjusted_loss

                    if model.bic < search_df['bic']:
                        search_df['bic'] = model.bic

                    if model.aic < search_df['aic']:
                        search_df['aic'] = model.aic

        print('Best values: ', search_df)

        return search_df['p'], search_df['d'], search_df['q']