from prophet import Prophet
from data.data_generator import DataLoader
from matplotlib import pyplot as plt


if __name__ == '__main__':
    dataloader= DataLoader(path_to_csv=r'')
    train, val = dataloader.load_data_for_article(1312)

    m = Prophet(interval_width=0.95, weekly_seasonality=False, daily_seasonality=False)
    m.fit(train)

    m = Prophet()
    m.fit(train)

    future = m.make_future_dataframe(periods=100)
    future.tail()

    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    fig1 = m.plot(forecast)
    plt.show()

    fig2 = m.plot_components(forecast)
    plt.show()

