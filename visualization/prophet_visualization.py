import matplotlib.pyplot as plt

from utils.path_handling import handle_parent_path


def plot_prophet_forecast(y_data, y_pred, filename, loss):
    fig, ax = plt.subplots()

    fig.set_size_inches((7, 3))

    ax.set_title(f'Prophet előrejelzés, RMSE = {loss}')

    ax.plot(y_data['ds'], y_data['y'], linewidth=1.5, label='Megfigyelés')
    ax.plot(y_pred['ds'], y_pred['yhat'], c='tab:orange', linewidth=1.5, label='Előrejelzés')

    ax.plot(y_pred['ds'], y_pred['yhat_lower'], c='tab:orange', alpha=0.5, linewidth=0.1)
    ax.plot(y_pred['ds'], y_pred['yhat_upper'], c='tab:orange', alpha=0.5, linewidth=0.1)
    ax.fill_between(y_pred['ds'], y_pred['yhat_lower'], y_pred['yhat_upper'], color='red', alpha=0.1,
                    label='0.05 konf. intervallum')
    ax.legend()

    handle_parent_path(filename)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
