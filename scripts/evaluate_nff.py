import argparse
import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from nff.analysis import loss_plot, plot_parity
from nff.data import Dataset, collate_dicts
from nff.train import evaluate, loss
from nff.train.builders.model import load_model
from nff.utils.cuda import cuda_devices_sorted_by_free_mem


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an NFF model on a dataset.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model",
    )
    parser.add_argument("--model_type", help="Name of model", type=str, default="CHGNetNFF")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data",
    )
    parser.add_argument(
        "--train_log_path",
        type=str,
        required=True,
        help="Path to the training log",
    )
    parser.add_argument(
        "--save_folder",
        type=Path,
        default="./",
        help="Folder to save output figures.",
    )
    parser.add_argument(
        "--plot_type",
        choices=["hexbin", "scatter"],
        default="hexbin",
        help="Type of plot to use",
    )
    parser.add_argument(
        "--per_atom_energy",
        action="store_true",
        help="Whether to calculate per atom energy",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="device to use for calculations",
    )

    return parser.parse_args()


def main(
    model_path: str,
    model_type: str,
    data_path: str,
    train_log_path: str,
    plot_type: str = "hexbin",
    per_atom_energy: bool = False,
    batch_size: int = 32,
    device: str = "cpu",
    save_folder: str = "./",
):
    """Evaluate an NFF model on a dataset.

    Args:
        model_path (str): path to the model
        model_type (str): name of the model
        data_path (str): path to the data
        train_log_path (str): path to the training log
        plot_type (str, optional): type of plot to use. Defaults to "hexbin".
        per_atom_energy (bool, optional): whether to calculate per atom energy. Defaults to False.
        batch_size (int, optional): batch size. Defaults to 32.
        device (str, optional): device to use. Defaults to "cpu".
        save_folder (str, optional): folder to save the results. Defaults to "./".
    """
    start_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    save_path = Path(save_folder)
    save_path.mkdir(exist_ok=True)

    # Determine the device to use
    if device not in "cpu" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if device == "cuda":
        # Determine cuda device with most available memory
        device_with_most_available_memory = cuda_devices_sorted_by_free_mem()[-1]
        device = f"cuda:{device_with_most_available_memory}"

    model = load_model(model_path, model_type=model_type, map_location=device)
    model.to(device)

    test_data = Dataset.from_file(data_path)

    units = model.units if hasattr(model, "units") else "eV"

    test_data.to_units(units)
    print(f"Using dataset units: {test_data.units}")

    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_dicts, pin_memory=True)

    loss_fn = loss.build_mse_loss(loss_coef={"energy": 0.05, "energy_grad": 1})

    print("Evaluating ...")
    results, targets, val_loss = evaluate(model, test_loader, loss_fn, device=device)

    # plot parity plot
    parity_plot_path = save_path / f"{start_time}_parity_plot"
    print(f"Saving parity plot to {parity_plot_path}")

    # convert units to per atom if needed
    if per_atom_energy and "/atom" not in test_data.units:
        units = f"{units}/atom"
        results["energy"] = [x / y for x, y in zip(results["energy"], targets["num_atoms"])]
        targets["energy"] = [x / y for x, y in zip(targets["energy"], targets["num_atoms"])]

    # Change energy_grad to force
    results["force"] = results["energy_grad"]
    targets["force"] = targets["energy_grad"]

    mae_energy, mae_force = plot_parity(
        results,
        targets,
        parity_plot_path,
        plot_type=plot_type,
        energy_key="energy",
        force_key="force",
        units={"force": "eV/Å", "energy": units},
    )

    # plot loss curves
    loss_plot_path = save_path / f"{start_time}_loss_plot"
    print(f"Saving loss plot to {loss_plot_path}")
    df = pd.read_csv(train_log_path)
    train_loss = df["Train loss"]
    val_loss = df["Validation loss"]
    total_loss = {"train": train_loss, "val": val_loss}
    loss_plot.plot_loss(total_loss, total_loss, loss_plot_path)


if __name__ == "__main__":
    args = parse_args()
    main(
        model_path=args.model_path,
        model_type=args.model_type,
        data_path=args.data_path,
        train_log_path=args.train_log_path,
        plot_type=args.plot_type,
        per_atom_energy=args.per_atom_energy,
        batch_size=args.batch_size,
        device=args.device,
        save_folder=args.save_folder,
    )
