from matplotlib import pyplot as plt


def plot_raw_predictions(model, raw_predictions, out_path_prefix):
    for idx in range(len(raw_predictions.x)):
        model.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
        plt.savefig(f'{out_path_prefix}_{idx}.png', dpi=600)
