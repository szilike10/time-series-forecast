import pandas as pd
from prophet import Prophet
from data.data_generator import DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from utils.path_handling import handle_parent_path


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
        if len(train) + len(val) >= 50:
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

            fig1 = m.plot(forecast)

            ax = plt.gca()
            ax.scatter(val['ds'], val['y'], c='red', s=10)

            # plt.show()
            filename = f'charts/{frequency}/{column}/categories/{category}.png'
            handle_parent_path(filename)
            plt.savefig(filename, dpi=300)

            # fig2 = m.plot_components(forecast)
            # plt.show()

    df_dict = {'category': list(mse_losses.keys()), 'mse_loss': mse_losses.values()}
    loss_df = pd.DataFrame.from_dict(df_dict)
    filename = f'charts/{frequency}/{column}/prophet_categories_losses.csv'
    handle_parent_path(filename)
    loss_df.to_csv(filename, index=False)


def plot_forecast(y_true, y_pred, frequency, column):
    fig, ax = plt.subplots()

    ax.plot(y_true['ds'], y_true['y'])
    ax.plot(y_pred['ds'], y_pred['yhat'], 'r')

    ax.plot(y_pred['ds'], y_pred['yhat_lower'], c='red', alpha=0.5, linewidth=1)
    ax.plot(y_pred['ds'], y_pred['yhat_upper'], c='red', alpha=0.5, linewidth=1)
    ax.fill_between(y_pred['ds'], y_pred['yhat_lower'], y_pred['yhat_upper'], color='red', alpha=0.1)

    filename = f'charts/{frequency}/{column}/prophet_forecast_combined_val.png'
    handle_parent_path(filename)
    plt.savefig(filename, dpi=300)


def predicts_combined_products(frequency='daily', column='valoare'):
    dataloader = DataLoader(path_to_csv=r'../../data/combined.csv')

    # train, val = dataloader.load_combined_data(column,
    #                                            start_date=pd.to_datetime('2022-01-01'),
    #                                            end_date=pd.to_datetime('2023-01-01'))

    train, val = dataloader.load_data(frequency=frequency,
                                      value_type=column,
                                      start_date=pd.to_datetime('2022-01-01'),
                                      end_date=pd.to_datetime('2023-01-01'))

    m = Prophet(interval_width=0.9, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
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

        fig1 = m.plot(forecast)

        ax = plt.gca()
        ax.scatter(val['ds'], val['y'], c='red', s=10)
        #
        # plt.show()

        filename = fr'charts/{frequency}/{column}/prophet_111.png'
        handle_parent_path(filename)
        plt.savefig(filename, dpi=300)
        #
        # fig2 = m.plot_components(forecast)
        # plt.show()

        loss_df = pd.DataFrame.from_dict({'name': [column], 'loss': [loss]})
        filename = f'charts/{frequency}/{column}/prophet_combined_.csv'
        handle_parent_path(filename)
        loss_df.to_csv(filename, index=False)

        plot_forecast(val, val_forecast, frequency, column)


if __name__ == '__main__':
    predict_all_categories(frequency='daily', column='valoare')
    # predict_all_cod_arts(frequency='daily', column='valoare')
    # predicts_combined_products(frequency='weekly', column='valoare')
