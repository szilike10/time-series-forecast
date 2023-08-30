import numpy as np
from matplotlib import pyplot as plt


def plot_raw_predictions(model, predictions, out_path_prefix, loss, quantiles=None):
    for idx in range(len(predictions.x["decoder_lengths"])):
        y_idx = len(quantiles) // 2 if quantiles is not None else 0

        y_past_time_idx = np.arange(predictions.x['encoder_lengths'].cpu()[idx])
        y_past = predictions.x['encoder_target'].cpu()[idx]

        y_future_time_idx = predictions.x['decoder_time_idx'].cpu()[idx, :]
        y_future = np.array(predictions.y[0].cpu()[idx, :])
        y_pred_future = predictions.output.prediction.cpu()[idx, :, y_idx]
        y_pred_low = predictions.output.prediction.cpu()[idx, :, 0]
        y_pred_high = predictions.output.prediction.cpu()[idx, :, -1]

        fig, ax = plt.subplots()
        ax.set_title(f'TFT előrejelzés, RMSE = {loss}')
        ax.plot(np.hstack([y_past_time_idx, y_future_time_idx]), np.hstack([y_past, y_future]), label='Megfigyelés')
        ax.plot(y_future_time_idx, y_pred_future, c='tab:orange', label='Elorejelzés')
        ax.plot(y_future_time_idx, y_pred_low, c='tab:orange', alpha=0.5, linewidth=0.1)
        ax.plot(y_future_time_idx, y_pred_high, c='tab:orange', alpha=0.5, linewidth=0.1)
        ax.fill_between(y_future_time_idx, y_pred_low, y_pred_high, color='red', alpha=0.15,
                        label='0.05 konf. intervallum')
        ax.legend()

        img_path = f'{out_path_prefix}_{idx}.png'
        print(img_path)
        fig.set_size_inches((7, 3.5))
        plt.tight_layout()
        plt.savefig(img_path, dpi=600)


        fig = model.plot_prediction(predictions.x, predictions.output, idx=idx)
        ax.set_title(f'TFT előrejelzés, RMSE = {loss}')
        img_path = f'{out_path_prefix}_{idx}_orig.png'
        print(img_path)
        fig.set_size_inches((7, 3.5))
        plt.tight_layout()
        plt.savefig(img_path, dpi=600)

        # fig = model.plot_prediction(predictions.y, predictions.output, idx=idx, add_loss_to_title=True)

        fig, ax = plt.subplots()
        ax.set_title(f'TFT előrejelzés, RMSE = {loss}')
        ax.plot(y_future_time_idx, y_future, label='Megfigyelés')
        ax.plot(y_future_time_idx, y_pred_future, c='tab:orange', label='Előrejelzés')
        ax.plot(y_future_time_idx, y_pred_low, c='tab:orange', alpha=0.5, linewidth=0.5)
        ax.plot(y_future_time_idx, y_pred_high, c='tab:orange', alpha=0.5, linewidth=0.5)
        ax.fill_between(y_future_time_idx, y_pred_low, y_pred_high, color='red', alpha=0.15,
                        label='0.05 konf. intervallum')
        ax.legend()

        img_path = f'{out_path_prefix}_{idx}_val.png'
        print(img_path)
        fig.set_size_inches((7, 3.5))
        plt.tight_layout()
        plt.savefig(img_path, dpi=600)
