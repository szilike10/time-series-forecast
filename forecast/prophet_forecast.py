from prophet import Prophet
from data.data_generator import DataLoader
from matplotlib import pyplot as plt
import pandas as pd


def predict_all_cod_arts():
    dataloader = DataLoader(path_to_csv=r'D:\Master\disszertacio\time-series-forecast\data\combined.csv')

    article_ids = dataloader.get_article_ids()

    # cod_art = 57
    # train, val = dataloader.load_data_for_article(1312)

    for cod_art in article_ids:
        train, val = dataloader.load_data_for_article(cod_art)

        m = Prophet(interval_width=0.80, weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        if len(train) > 1:
            m.fit(train)

            future = m.make_future_dataframe(periods=100)
            future.tail()

            future = train.append(val)

            forecast = m.predict(future)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            fig1 = m.plot(forecast)

            # plt.show()
            filename = fr'D:\Master\disszertacio\time-series-forecast\data\predictions\articles\filled\daily_{cod_art}.png'
            plt.savefig(filename, dpi=300)

            # fig2 = m.plot_components(forecast)
            # plt.show()


def predict_all_categories():
    dataloader = DataLoader(path_to_csv=r'D:\Master\disszertacio\time-series-forecast\data\combined.csv')

    categories = dataloader.get_categories()

    # cod_art = 57
    # train, val = dataloader.load_data_for_article(1312)

    for category in categories:
        train, val = dataloader.load_data_for_category(category, fill_missing_data=False)

        m = Prophet(interval_width=0.80, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        if len(train) + len(val) >= 50:
            m.fit(train)

            future = m.make_future_dataframe(periods=100)
            future.tail()
            #
            # future = train.append(val)

            forecast = m.predict(future)
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
            fig1 = m.plot(forecast)

            # plt.show()
            # filename = fr'D:\Master\disszertacio\time-series-forecast\data\predictions\categories\filled\daily_{category}.png'
            # plt.savefig(filename, dpi=300)
            #
            # fig2 = m.plot_components(forecast)
            plt.show()


if __name__ == '__main__':
    predict_all_categories()
