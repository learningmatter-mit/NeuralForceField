import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from nff.data import Dataset, collate_dicts
from nff.nn.models.chgnet import CHGNetNFF
from nff.train import Trainer, get_model, hooks, loss, metrics
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from nff.utils.misc import log

torch.set_printoptions(precision=3, sci_mode=False)

logger = logging.getLogger(__name__)


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Name and seed
    parser.add_argument("--name", help="experiment name", required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=1337)

    # Directories
    parser.add_argument(
        "--train_dir",
        help="directory for training",
        type=str,
        default="/mnt/data0/dux/nff_working/finetuning/training",
    )

    parser.add_argument(
        "--r_max", help="distance cutoff (in Ang)", type=float, default=6.0
    )

    # Dataset
    parser.add_argument(
        "--train_file", help="Training set pth.tar file", type=str, required=True
    )
    parser.add_argument(
        "--val_file", help="Validation set pth.tar file", type=str, required=False
    )
    parser.add_argument(
        "--test_file",
        help="Test set pth.tar file",
        type=str,
    )

    # Model params
    parser.add_argument(
        "--model_type", help="Name of model", type=str, default="CHGNetNFF"
    )
    parser.add_argument(
        "--model_params_path", help="Path to model parameters", type=str
    )
    parser.add_argument(
        "--fine_tune",
        help="Whether to fine-tune the model",
        action="store_true",
    )

    # Training params
    parser.add_argument(
        "--forces_weight", help="weight of forces loss", type=float, default=1.0
    )
    parser.add_argument(
        "--energy_weight", help="weight of energy loss", type=float, default=0.05
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=16)

    parser.add_argument(
        "--lr", help="Learning rate of optimizer", type=float, default=0.01
    )
    parser.add_argument(
        "--max_num_epochs", help="Maximum number of epochs", type=int, default=500
    )
    parser.add_argument(
        "--patience",
        help="Maximum number of consecutive epochs of increasing loss",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--allow_grad",
        help="Which layers to fine-tune",
        choices=["ALL", "LAST"],
        default="LAST",
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers for data loader",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pin_memory",
        help="Whether to pin memory for data loader",
        action="store_true",
    )
    parser.add_argument(
        "--targets",
        default="ef",
        help="Which targets to predict",
    )
    parser.add_argument(
        "--criterion",
        default="MSE",
        help="Loss function",
        choices=["MSE", "MAE", "Huber"],
    )
    return parser


def log_train(msg):
    log("train", msg)


def main(all_params):
    # Set seeds
    torch.manual_seed(all_params.seed)
    np.random.seed(all_params.seed)

    # Create directory for saving model and logs
    save_path = os.path.join(all_params.train_dir, all_params.name)
    os.makedirs(save_path, exist_ok=True)

    # Determine the device to use
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if device == "cuda":
        # Determine cuda device with most available memory
        device_with_most_available_memory = cuda_devices_sorted_by_free_mem()[-1]
        device = f"cuda:{device_with_most_available_memory}"

    trainingparams = {
        "targets": "ef",
        "lr": 0.001,
        "weight_decay": 0.0,
        "criterion": "MSE",
        "energy_loss_ratio": 1,
        "force_loss_ratio": 1,
        "stress_loss_ratio": 0.1,
        "mag_loss_ratio": 0.1,
        "delta_huber": 0.1,
        "is_intensive": True,
        "n_epochs": all_params.max_num_epochs,
        "scheduler_decay_fraction": 1e-2,
        "batch_size": all_params.batch_size,
        "workers": all_params.num_workers,
        "device": device,
    }

    modelparams = {
        "atom_fea_dim": 64,
        "bond_fea_dim": 64,
        "angle_fea_dim": 64,
        "composition_model": "MPtrj",
        "num_radial": 31,
        "num_angular": 31,
        "n_conv": 4,
        "atom_conv_hidden_dim": 64,
        "update_bond": True,
        "bond_conv_hidden_dim": 64,
        "update_angle": True,
        "angle_layer_hidden_dim": 0,
        "conv_dropout": 0.0,
        "read_out": "ave",
        "gMLP_norm": "layer",
        "readout_norm": "layer",
        "mlp_hidden_dims": [64, 64, 64],
        "mlp_first": True,
        "is_intensive": True,
        "non_linearity": "silu",
        "atom_graph_cutoff": all_params.r_max,
        "bond_graph_cutoff": 3.0,
        "graph_converter_algorithm": "fast",
        "cutoff_coeff": 8,
        "learnable_rbf": True,
        "device": device,
    }

    if all_params.fine_tune:
        logger.info("Fine-tuning model")
        model = CHGNetNFF.load(device=device)
        # Optionally fix the weights of some layers
        if all_params.allow_grad == "ALL":
            for layer in [
                model.atom_embedding,
                model.bond_embedding,
                model.angle_embedding,
                model.bond_basis_expansion,
                model.angle_basis_expansion,
                model.atom_conv_layers[:],
                model.bond_conv_layers,
                model.angle_layers,
            ]:
                for param in layer.parameters():
                    param.requires_grad = True
        elif all_params.allow_grad == "LAST":
            for layer in [
                model.atom_embedding,
                model.bond_embedding,
                model.angle_embedding,
                model.bond_basis_expansion,
                model.angle_basis_expansion,
                model.atom_conv_layers[:-1],
                model.bond_conv_layers,
                model.angle_layers,
            ]:
                for param in layer.parameters():
                    param.requires_grad = False
            for param in model.atom_conv_layers[-1].parameters():
                param.requires_grad = True
    else:
        model = get_model(modelparams, model_type=all_params.model_type)

    model.to(device)
    model.float()

    logger.info(
        "Num trainable parameters: %s",
        sum([p.numel() for p in model.parameters() if p.requires_grad]),
    )

    # Datasets without per-atom offsets etc.
    train = Dataset.from_file(all_params.train_file)
    val = Dataset.from_file(all_params.val_file)
    test = Dataset.from_file(all_params.test_file)
    train.to_units(model.units)
    val.to_units(model.units)
    test.to_units(model.units)

    train_loader = DataLoader(
        train,
        batch_size=trainingparams["batch_size"],
        num_workers=trainingparams["workers"],
        collate_fn=collate_dicts,
    )

    val_loader = DataLoader(
        val,
        batch_size=trainingparams["batch_size"],
        num_workers=trainingparams["workers"],
        collate_fn=collate_dicts,
    )

    # define loss
    loss_fn = loss.build_mse_loss(
        loss_coef={
            "energy": all_params.energy_weight,
            "energy_grad": all_params.forces_weight,
        }
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainingparams["lr"],
        weight_decay=trainingparams["weight_decay"],
    )

    # monitor energy and force MAE
    train_metrics = [
        metrics.MeanAbsoluteError("energy"),
        metrics.MeanAbsoluteError("energy_grad"),
    ]

    train_hooks = [
        hooks.WarmRestartHook(
            T0=trainingparams["n_epochs"],
            Tmult=1,
            lr_min=trainingparams["scheduler_decay_fraction"],
            lr_factor=trainingparams["lr"],
            optimizer=optimizer,
        ),
        hooks.MaxEpochHook(all_params.max_num_epochs),
        hooks.CSVHook(
            save_path,
            metrics=train_metrics,
        ),
        hooks.PrintingHook(
            save_path,
            metrics=train_metrics,
            separator=" | ",
            time_strf="%M:%S",
        ),
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer,
            patience=all_params.patience,
            factor=trainingparams["lr"],
            min_lr=1e-7,
            window_length=1,
            stop_after_min=True,
        ),
    ]

    # Perform actual training
    T = Trainer(
        model_path=save_path,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        checkpoint_interval=1,
        hooks=train_hooks,
    )

    T.train(device=device, n_epochs=trainingparams["n_epochs"])

    log_train("model saved in " + save_path)


if __name__ == "__main__":
    args = build_default_arg_parser().parse_args()
    main(args)
