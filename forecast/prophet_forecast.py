from prophet import Prophet
from data.data_generator import DataLoader
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error


def predict_all_cod_arts():
    dataloader = DataLoader(path_to_csv=r'C:\Users\bas6clj\time-series-forecast\data\combined.csv')

    article_ids = dataloader.get_article_ids()

    # cod_art = 57
    # train, val = dataloader.load_data_for_article(1312)

    mse_losses = {}

    for cod_art in article_ids:
        train, val = dataloader.load_data_for_article(cod_art, fill_missing_data=False)

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

            # fig1 = m.plot(forecast)

            # plt.show()
            # filename = fr'C:\Users\bas6clj\time-series-forecast\data\predictions\articles\not_filled\daily_{cod_art}.png'
            # plt.savefig(filename, dpi=300)

            # fig2 = m.plot_components(forecast)
            # plt.show()

    df_dict = {'cod_art': list(mse_losses.keys()), 'mse_loss': mse_losses.values()}
    loss_df = pd.DataFrame.from_dict(df_dict)
    loss_df.to_csv('prophet_cod_art_losses.csv', index=False)

def predict_all_categories():
    dataloader = DataLoader(path_to_csv=r'C:\Users\bas6clj\time-series-forecast\data\combined.csv')

    categories = dataloader.get_categories()

    mse_losses = {}

    for category in categories:
        train, val = dataloader.load_data_for_category(category, fill_missing_data=False)

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

            # fig1 = m.plot(forecast)
            #
            # ax = plt.gca()
            # ax.scatter(val['ds'], val['y'], c='red', s=10)

            # plt.show()
            # filename = fr'C:\Users\bas6clj\time-series-forecast\data\predictions\categories\not_filled\daily_{category}.png'
            # plt.savefig(filename, dpi=300)
            #
            # fig2 = m.plot_components(forecast)
            # plt.show()

    df_dict = {'category': list(mse_losses.keys()), 'mse_loss': mse_losses.values()}
    loss_df = pd.DataFrame.from_dict(df_dict)
    loss_df.to_csv('prophet_categories_losses.csv', index=False)


def plot_forecast(y_true, y_pred, column=''):
    fig, ax = plt.subplots()

    ax.plot(y_true['ds'], y_true['y'])
    ax.plot(y_pred['ds'], y_pred['yhat'], 'r')

    plt.savefig(f'prophet_forecast_combined_{column}', dpi=300)

def predicts_combined_products():
    dataloader = DataLoader(path_to_csv=r'C:\Users\bas6clj\time-series-forecast\data\combined.csv')
    column = 'cantitate'

    train, val = dataloader.load_combined_data(column,
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

        filename = fr'C:\Users\bas6clj\time-series-forecast\data\predictions\daily_{column}_111.png'
        plt.savefig(filename, dpi=300)
        #
        # fig2 = m.plot_components(forecast)
        # plt.show()

        loss_df = pd.DataFrame.from_dict({'name': [column], 'loss': [loss]})
        loss_df.to_csv(f'prophet_combined_{column}.csv', index=False)

        plot_forecast(val, val_forecast, column)


if __name__ == '__main__':
    # predict_all_categories()
    # predict_all_cod_arts()
    predicts_combined_products()
