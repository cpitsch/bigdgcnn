from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--num_models", "-n", type=int, default=10)
parser.add_argument("--num_cores", "-c", type=int, help="Number of cores to use for multiprocessing", default=-1)
lr_groups = parser.add_mutually_exclusive_group(required=True)
lr_groups.add_argument("--1e-4", action="store_true", help="Use a learning rate of 1e-4", dest="lr_1e_4")
lr_groups.add_argument("--1e-3", action="store_true", help="Use a learning rate of 1e-3", dest="lr_1e_3")
args = parser.parse_args()

from typing import List
from bigdgcnn.ml.model import BIG_DGCNN
import pm4py
from bigdgcnn.util import print_log_statistics
from torch.multiprocessing import Pool, cpu_count, freeze_support
from tqdm.auto import tqdm
from statistics import mean, stdev
from timeit import default_timer
from datetime import timedelta
from pathlib import Path
from tabulate import tabulate
import pandas as pd

def train_model(args) -> BIG_DGCNN:
    _, log, lr = args
    model = BIG_DGCNN(
        sort_pooling_k=5,
        layer_sizes=[32, 32, 32, 32, 32],
        batch_size=32,
        learning_rate=lr,
        dropout_rate=0.2,
        sizes_1d_convolutions=[32],
        dense_layer_sizes=[32],
        epochs=100,
        use_cuda_if_available=False
    )
    model.train(log, "bpi12")
    return model


def main():
    NUM_MODELS = args.num_models

    log = pm4py.read_xes(r"./Event Logs/BPI_Challenge_2012_W.xes.gz")
    print_log_statistics(log)

    LEARNING_RATE = 1e-4 if args.lr_1e_4 else 1e-3
    out_path = Path("./Experiments/BPI12/", "LR_1e-4" if args.lr_1e_4 else "LR_1e-3")
    if not out_path.exists():
        out_path.mkdir(parents=True)

    start_time = default_timer()

    cpu_count = cpu_count() -2 if args.num_cores == -1 else args.num_cores
    freeze_support()
    with Pool(cpu_count) as p:
        models: List[BIG_DGCNN] = p.map(train_model, tqdm([(idx, log, LEARNING_RATE) for idx in range(NUM_MODELS)], desc="Training Models"))

    accuracies = [model.test_accuracy for model in models]
    print(accuracies)
    print(f"Number of Models: {len(accuracies)}")
    print(f"Mean Accuracy: {mean(accuracies)}")
    print(f"Standard Deviation: {stdev(accuracies)}")
    print(f"Total Time: {timedelta(seconds=default_timer() - start_time)}")

    for idx, model in enumerate(models, start=1):
        model.save(Path(out_path, f"BPI12_{idx}.pt")) # Save the parameters of the internal pytorch model
        model.plot_training_history().savefig(Path(out_path, f"BPI12_{idx}.pdf"))

    _model0 = models[0]
    model_config = {
        "sort_pooling_k": _model0.sort_pooling_k,
        "layer_sizes": _model0.layer_sizes,
        "batch_size": _model0.batch_size,
        "learning_rate": _model0.learning_rate,
        "dropout_rate": _model0.dropout_rate,
        "sizes_1d_convolutions": _model0.sizes_1d_convolutions,
        "dense_layer_sizes": _model0.dense_layer_sizes,
        "epochs": _model0.epochs
    }

    
    # Write results to file
    with open(Path(out_path, "summary.txt"), "w") as f:
        f.write(f"Number of Models: {len(accuracies)}\n")
        f.write(f"Test Accuracies: {accuracies}\n")
        f.write(f"Mean Accuracy: {mean(accuracies)}\n")
        f.write(f"Standard Deviation: {stdev(accuracies)}\n")
        f.write(f"Total Time: {timedelta(seconds=default_timer() - start_time)}\n\n")
        f.write("\n\n")

        f.write("Model Config:\n")
        f.write(tabulate(model_config.items(), tablefmt="github"))
        f.write("\n\n")
        f.write("\n\n")

        for model_number, model in enumerate(models, start=1):
            f.write("#"*10 + f" Model {model_number} " + "#"*10 + "\n\n")
            f.write(f"Test Accuracy: {model.test_accuracy}\n\n")
            # print empty line
            f.write("\n")
            # The range length is the number of epochs
            data = zip(range(1,len(model.train_losses)+1), model.train_losses, model.validation_losses, model.train_accuracies, model.validation_accuracies)
            headers = ["Epoch", "Train Loss", "Validation Loss", "Train Accuracy", "Validation Accuracy"]
            f.write(tabulate(data, headers=headers, tablefmt="github"))
            f.write("\n\n")

    # Save as a csv
    pd.DataFrame([
        {
            "Model": idx,
            "Epoch": epoch,
            "Train Loss": model.train_losses[epoch-1],
            "Validation Loss": model.validation_losses[epoch-1],
            "Train Accuracy": model.train_accuracies[epoch-1],
            "Validation Accuracy": model.validation_accuracies[epoch-1],
            "Test Accuracy": model.test_accuracy if epoch == model.epochs else None
        }
        for idx, model in enumerate(models, start=1)
        for epoch in range(1, model.epochs + 1)
    ]).to_csv(Path(out_path, "training_histories.csv"), index=False)
if __name__ == '__main__':
        main()
