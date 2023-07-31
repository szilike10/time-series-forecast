import pandas as pd
from prophet import Prophet
from data.data_generator import DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from utils.path_handling import handle_parent_path
import numpy as np

from visualization.prophet_visualization import plot_prophet_forecast


def predict_all_cod_arts(frequency='daily', column='valoare'):
    dataloader = DataLoader(path_to_csv=r'../../data/combined.csv')

    article_ids = dataloader.get_article_ids()

    # cod_art = 57
    # train, val = dataloader.load_data_for_article(1312)

    mse_losses = {}

    for cod_art in article_ids:
        train, val = dataloader.load_data(frequency=frequency,
                                          item_type='cod_art',
                                          type_identifier=cod_art,
                                          value_type=column,
                                          start_date=pd.to_datetime('2022-01-01'),
                                          end_date=pd.to_datetime('2023-01-01'))

        m = Prophet(interval_width=0.80, weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=True)
        if len(train) >= 50:
            m.fit(train)

            future = m.make_future_dataframe(periods=100)
            future.tail()

            # future = train.append(val)

            val_forecast = m.predict(val)
            loss = mean_squared_error(val['y'], val_forecast['yhat'])
            mse_losses[cod_art] = loss

            forecast = m.predict(future)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

            fig1 = m.plot(forecast)

            ax = plt.gca()
            ax.scatter(val['ds'], val['y'], c='red', s=10)

            # plt.show()
            filename = f'charts/{frequency}/{column}/articles/{cod_art}.png'
            plt.savefig(filename, dpi=300)

            # fig2 = m.plot_components(forecast)
            # plt.show()

    df_dict = {'cod_art': list(mse_losses.keys()), 'mse_loss': mse_losses.values()}
    loss_df = pd.DataFrame.from_dict(df_dict)
    filename = f'charts/{frequency}/{column}/prophet_cod_art_losses.csv'
    handle_parent_path(filename)
    loss_df.to_csv(filename, index=False)


def predict_all_categories(frequency='daily', column='valoare'):
    dataloader = DataLoader(path_to_csv=r'../../data/combined.csv')

    categories = dataloader.get_categories()

    mse_losses = {}

    for category in categories:
        train, val = dataloader.load_data(frequency=frequency,
                                          item_type='category',
                                          type_identifier=category,
                                          value_type=column,
                                          start_date=pd.to_datetime('2022-01-01'),
                                          end_date=pd.to_datetime('2023-01-01'))

        m = Prophet(interval_width=0.80, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        if len(train) + len(val) >= 100:
            m.fit(train)

            future = m.make_future_dataframe(periods=100)
            future.tail()
            #
            # future = train.append(val)

            val_forecast = m.predict(val)
            loss = mean_squared_error(val['y'], val_forecast['yhat'])
            mse_losses[category] = loss

            forecast = m.predict(future)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

            # fig1 = m.plot(forecast)
            #
            # ax = plt.gca()
            # ax.scatter(val['ds'], val['y'], c='red', s=10)

            # plt.show()
            filename = f'charts/{frequency}/{column}/categories/{category}.png'
            handle_parent_path(filename)
            plot_prophet_forecast(pd.concat([train, val]), val_forecast, filename)
            # plt.savefig(filename, dpi=300)

            # fig2 = m.plot_components(forecast)
            # plt.show()

    df_dict = {'category': list(mse_losses.keys()), 'mse_loss': mse_losses.values()}
    loss_df = pd.DataFrame.from_dict(df_dict)
    filename = f'charts/{frequency}/{column}/prophet_category_losses.csv'
    handle_parent_path(filename)
    loss_df.to_csv(filename, index=False)


def predict_combined_products(frequency='daily', column='valoare'):
    dataloader = DataLoader(path_to_csv=r'../../data/combined.csv')

    # train, val = dataloader.load_combined_data(column,
    #                                            start_date=pd.to_datetime('2022-01-01'),
    #                                            end_date=pd.to_datetime('2023-01-01'))

    train, val = dataloader.load_data(frequency=frequency,
                                      value_type=column,
                                      start_date=pd.to_datetime('2022-01-01'),
                                      end_date=pd.to_datetime('2023-01-01'))

    m = Prophet(interval_width=0.9, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    if len(train) + len(val) >= 50:
        m.fit(train)

        future = m.make_future_dataframe(periods=500)
        future.tail()
        #
        future = train.append(val)

        forecast = m.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

        val_forecast = m.predict(val)
        loss = mean_squared_error(val['y'], val_forecast['yhat'])
        print(loss)

        # fig1 = m.plot(forecast)



        # ax = plt.gca()
        # ax.scatter(val['ds'], val['y'], c='red', s=10)
        #
        # plt.show()

        # filename = fr'charts/{frequency}/{column}/prophet_011.png'
        # handle_parent_path(filename)
        # plt.savefig(filename, dpi=300)
        #
        # fig2 = m.plot_components(forecast)
        # plt.show()

        loss_df = pd.DataFrame.from_dict({'name': [column], 'loss': [loss]})
        filename = f'charts/{frequency}/{column}/prophet_combined_.csv'
        handle_parent_path(filename)
        loss_df.to_csv(filename, index=False)

        plot_filename = f'charts/{frequency}/{column}/prophet_forecast_combined_val.png'
        plot_prophet_forecast(val, val_forecast, plot_filename)
        plot_filename = f'charts/{frequency}/{column}/prophet_forecast_combined.png'
        plot_prophet_forecast(pd.concat([train, val]), val_forecast, plot_filename)


def compare_to_mean(item_type=None, frequency='daily', column='valoare'):
    dataloader = DataLoader(path_to_csv=r'../../data/combined.csv')

    mse_loss = []
    prophet_losses = []
    diffs = []
    percentages = []
    improved = []
    relevant_type_identifiers = []

    type_identifiers = [None]
    if item_type == 'category':
        type_identifiers = dataloader.get_categories()
    elif item_type == 'cod_art':
        type_identifiers = dataloader.get_article_ids()

    df = pd.read_csv(f'charts/{frequency}/{column}/prophet_{str(item_type)}_losses.csv', index_col=str(item_type))

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
            prophet_loss = df.loc[type_identifier, 'mse_loss']
            diff = prophet_loss - loss

            improved.append(1 if diff < 0 else -1)
            percentages.append(-diff / loss)

            mse_loss.append(loss)
            prophet_losses.append(prophet_loss)
            diffs.append(diff)

    df = pd.DataFrame.from_dict({
        str(item_type): relevant_type_identifiers,
        'mse_loss_of_mean': mse_loss,
        'prophet_loss': prophet_losses,
        'diff': diffs,
        'improved': improved,
        'improved %': percentages
    })
    filename = f'charts/comparison_to_mean/{frequency}/{column}/prophet_category_compare_to_mean.csv'
    handle_parent_path(filename)
    df.to_csv(filename, index=False)




if __name__ == '__main__':
    predict_all_categories(frequency='daily', column='valoare')
    # predict_all_cod_arts(frequency='daily', column='valoare')
    # predict_combined_products(frequency='weekly', column='valoare')
    # compare_to_mean(item_type='category', frequency='daily', column='valoare')
