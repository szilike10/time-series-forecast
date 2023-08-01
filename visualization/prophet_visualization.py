import matplotlib.pyplot as plt

from utils.path_handling import handle_parent_path


def plot_prophet_forecast(y_data, y_pred, filename):
    fig, ax = plt.subplots()

    fig.set_size_inches((7, 3))

    ax.plot(y_data['ds'], y_data['y'], linewidth=1)
    ax.plot(y_pred['ds'], y_pred['yhat'], 'r', linewidth=1)

    ax.plot(y_pred['ds'], y_pred['yhat_lower'], c='red', alpha=0.5, linewidth=0.5)
    ax.plot(y_pred['ds'], y_pred['yhat_upper'], c='red', alpha=0.5, linewidth=0.5)
    ax.fill_between(y_pred['ds'], y_pred['yhat_lower'], y_pred['yhat_upper'], color='red', alpha=0.1)

    handle_parent_path(filename)
    plt.savefig(filename, dpi=300)
