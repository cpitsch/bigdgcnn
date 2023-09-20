from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--num_models", "-n", type=int, default=10)
parser.add_argument("--num_cores", "-c", type=int, help="Number of cores to use for multiprocessing", default=-1)
args = parser.parse_args()

from typing import List
from bigdgcnn.ml.model_pytorch import BIG_DGCNN
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
    _, log = args
    model = BIG_DGCNN(
        sort_pooling_k=30,
        layer_sizes=[32, 32, 32, 32, 32],
        batch_size=32,
        learning_rate=1e-3,
        dropout_rate=0.1,
        sizes_1d_convolutions=[32],
        dense_layer_sizes=[32],
        epochs=100,
        use_cuda_if_available=False
    )
    model.train(log, "helpdesk")
    return model


def main():
    NUM_MODELS = args.num_models

    log = pm4py.read_xes(r"./Event Logs/helpdesk.xes.gz")
    print_log_statistics(log)

    out_path = Path("./Experiments/Helpdesk/")
    if not out_path.exists():
        out_path.mkdir(parents=True)

    start_time = default_timer()
    freeze_support()
    num_cores = cpu_count() -2 if args.num_cores == -1 else args.num_cores
    print(f"Multiprocessing using {num_cores} cores.")
    with Pool(num_cores) as p:
        models: List[BIG_DGCNN] = p.map(train_model, tqdm([(idx, log) for idx in range(NUM_MODELS)], desc="Training Models"))

    accuracies = [model.test_accuracy for model in models]
    print(accuracies)
    print(f"Number of Models: {len(accuracies)}")
    print(f"Mean Accuracy: {mean(accuracies)}")
    print(f"Standard Deviation: {stdev(accuracies)}")
    print(f"Total Time: {timedelta(seconds=default_timer() - start_time)}")

    for idx, model in enumerate(models, start=1):
        model.save(Path(out_path, f"Helpdesk_{idx}.pt").as_posix()) # Save the parameters of the internal pytorch model
        model.plot_training_history().savefig(Path(out_path, f"Helpdesk_{idx}.pdf").as_posix())

    # Write results to file
    with open(Path(out_path, "summary.txt"), "w") as f:
        f.write(f"Number of Models: {len(accuracies)}\n")
        f.write(f"Test Accuracies: {accuracies}\n")
        f.write(f"Mean Accuracy: {mean(accuracies)}\n")
        f.write(f"Standard Deviation: {stdev(accuracies)}\n")
        f.write(f"Total Time: {timedelta(seconds=default_timer() - start_time)}\n\n")
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
