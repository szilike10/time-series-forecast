import matplotlib.pyplot as plt

from utils.path_handling import handle_parent_path


def plot_forecast(y_true, y_pred, filename, loss, draw_plot=False):
    fig, ax = plt.subplots()

    fig.set_size_inches((7, 3))
    plt.tight_layout()

    ax.set_title(f'Statsmodels előrejelzés, RMSE = {loss}')

    ax.plot(y_true['ds'], y_true['y'], linewidth=1.5, label='Megfigyelés')
    ax.plot(y_pred['ds'], y_pred['y'], 'tab:orange', linewidth=1.5, label='Előrejelzés')

    ax.plot(y_pred['ds'], y_pred['lower y'], c='tab:orange', alpha=0.5, linewidth=0.1)
    ax.plot(y_pred['ds'], y_pred['upper y'], c='tab:orange', alpha=0.5, linewidth=0.1)
    ax.fill_between(y_pred['ds'], y_pred['lower y'], y_pred['upper y'], color='red', alpha=0.1,
                    label='0.05 konf. intervallum')
    ax.legend()

    handle_parent_path(filename)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

    if draw_plot:
        plt.show()