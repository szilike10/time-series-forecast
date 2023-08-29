from matplotlib import pyplot as plt


def plot_raw_predictions(model, predictions, out_path_prefix):
    for idx in range(len(predictions.x["decoder_lengths"])):
        fig = model.plot_prediction(predictions.x, predictions.output, idx=idx, add_loss_to_title=True)
        img_path = f'{out_path_prefix}_{idx}.png'
        print(img_path)
        fig.set_size_inches((7, 3.5))
        plt.tight_layout()
        plt.savefig(img_path, dpi=600)

        fig = model.plot_prediction(predictions.y, predictions.output, idx=idx, add_loss_to_title=True)
        img_path = f'{out_path_prefix}_{idx}_val.png'
        print(img_path)
        fig.set_size_inches((7, 3.5))
        plt.tight_layout()
        plt.savefig(img_path, dpi=600)
