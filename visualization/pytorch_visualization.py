import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.path_handling import handle_parent_path


def plot_raw_predictions(model, predictions, out_path_prefix, loss, start_date, frequency, quantiles=None):
    offset_multiplier = 1
    if frequency == 'weekly':
        offset_multiplier = 7

    for idx in range(len(predictions.x["decoder_lengths"])):
        y_idx = len(quantiles) // 2 if quantiles is not None else 0

        y_past_time_idx = np.arange(predictions.x['encoder_lengths'].cpu()[idx])
        y_past_ds = [start_date + pd.DateOffset(offset * offset_multiplier) for offset in y_past_time_idx]
        y_past = predictions.x['encoder_target'].cpu()[idx]

        y_future_time_idx = predictions.x['decoder_time_idx'].cpu()[idx, :]
        y_future_ds = [start_date + pd.DateOffset(offset * offset_multiplier) for offset in y_future_time_idx]
        y_future = np.array(predictions.y[0].cpu()[idx, :])
        y_pred_future = predictions.output.prediction.cpu()[idx, :, y_idx]
        y_pred_low = predictions.output.prediction.cpu()[idx, :, 0]
        y_pred_high = predictions.output.prediction.cpu()[idx, :, -1]

        fig, ax = plt.subplots()
        ax.set_title(f'TFT előrejelzés, RMSE = {loss}')
        y_all_ds = y_past_ds + y_future_ds
        ax.plot(y_all_ds, np.hstack([y_past, y_future]), label='Megfigyelés')
        ax.plot(y_future_ds, y_pred_future, c='tab:orange', label='Elorejelzés')
        ax.plot(y_future_ds, y_pred_low, c='tab:orange', alpha=0.5, linewidth=0.1)
        ax.plot(y_future_ds, y_pred_high, c='tab:orange', alpha=0.5, linewidth=0.1)
        ax.fill_between(y_future_ds, y_pred_low, y_pred_high, color='red', alpha=0.15,
                        label='0.05 konf. intervallum')
        ax.legend()

        img_path = f'{out_path_prefix}/{frequency}_{idx}.png'
        handle_parent_path(img_path)
        print(img_path)
        fig.set_size_inches((7, 3))
        plt.tight_layout()
        plt.savefig(img_path, dpi=600)


        # fig = model.plot_prediction(predictions.x, predictions.output, idx=idx)
        # ax.set_title(f'TFT előrejelzés, RMSE = {loss}')
        # img_path = f'{out_path_prefix}/{frequency}_{idx}_orig.png'
        # handle_parent_path(img_path)
        # print(img_path)
        # fig.set_size_inches((7, 3))
        # # plt.tight_layout()
        # plt.savefig(img_path, dpi=600)

        fig, ax = plt.subplots()
        ax.set_title(f'TFT előrejelzés, RMSE = {loss}')
        ax.plot(y_future_ds, y_future, label='Megfigyelés')
        ax.plot(y_future_ds, y_pred_future, c='tab:orange', label='Előrejelzés')
        ax.plot(y_future_ds, y_pred_low, c='tab:orange', alpha=0.5, linewidth=0.5)
        ax.plot(y_future_ds, y_pred_high, c='tab:orange', alpha=0.5, linewidth=0.5)
        ax.fill_between(y_future_ds, y_pred_low, y_pred_high, color='red', alpha=0.1,
                        label='0.05 konf. intervallum')
        ax.legend()

        img_path = f'{out_path_prefix}/{frequency}_{idx}_val.png'
        handle_parent_path(img_path)
        print(img_path)
        fig.set_size_inches((7, 3))
        plt.tight_layout()
        plt.savefig(img_path, dpi=600)
