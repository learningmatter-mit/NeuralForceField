import argparse
import logging
from pathlib import Path
from typing import Iterable, Literal, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from nff.data import Dataset, collate_dicts
from nff.io.mace import update_mace_init_params
from nff.train import Trainer, get_model, hooks, load_model, loss, metrics
from nff.train.loss import mae_operation, mse_operation
from nff.utils.cuda import cuda_devices_sorted_by_free_mem
from nff.utils.misc import log

torch.set_printoptions(precision=3, sci_mode=False)

logger = logging.getLogger(__name__)

LOSS_OPERATIONS = {"MAE": mae_operation, "MSE": mse_operation}


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Name and seed
    parser.add_argument("--name", help="experiment name", required=True)
    parser.add_argument("--seed", help="random seed", type=int, default=1337)

    # Model params
    parser.add_argument("--model_type", help="Name of model", type=str, default="CHGNetNFF")
    parser.add_argument("--model_params_path", help="Path to model parameters", type=str)
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to a trained model",
    )
    parser.add_argument(
        "--fine_tune",
        help="Whether to fine-tune the model",
        action="store_true",
    )

    # Dataset
    parser.add_argument("--train_file", help="Training set pth.tar file", type=str, required=True)
    parser.add_argument("--val_file", help="Validation set pth.tar file", type=str, required=False)
    parser.add_argument(
        "--test_file",
        help="Test set pth.tar file",
        type=str,
    )

    # Training params
    parser.add_argument(
        "--train_dir",
        help="Path to store training results",
        type=str,
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=str,
        default=["energy", "energy_grad"],
        help="Which targets to predict",
    )
    parser.add_argument(
        "--loss_weights",
        nargs="+",
        type=float,
        default=[0.05, 1.0],
        help="Loss weight of each target",
    )
    parser.add_argument(
        "--criterion",
        default="MSE",
        help="Loss function",
        choices=["MSE", "MAE", "Huber"],  # TODO, build huber loss
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=16)
    parser.add_argument("--lr", help="Starting learning rate of optimizer", type=float, default=1e-3)
    parser.add_argument("--min_lr", help="Minimum rate of optimizer", type=float, default=1e-6)
    parser.add_argument("--max_num_epochs", help="Maximum number of epochs", type=int, default=500)
    parser.add_argument(
        "--patience",
        help="Maximum number of consecutive epochs of increasing loss",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--weight_decay",
        help="Fraction to reduce weights by at each step",
        type=float,
        default=0.0,
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
    return parser


def log_train(msg):
    pass
    log("train", msg)


def main(
    name: str,
    model_type: str,
    model_params_path: Union[str, Path],
    model_path: Union[str, Path],
    train_dir: Union[str, Path],
    train_file: Union[str, Path],
    val_file: Union[str, Path],
    fine_tune: bool = False,
    allow_grad: Literal["ALL", "LAST"] = "LAST",
    targets: Iterable[str] = ["energy", "energy_grad"],
    loss_weights: Iterable[float] = [0.05, 1.0],
    criterion: Literal["MSE", "MAE"] = "MSE",
    batch_size: int = 16,
    lr: float = 1e-3,
    min_lr: float = 1e-6,
    max_num_epochs: int = 200,
    patience: int = 25,
    weight_decay: float = 0.0,
    num_workers: int = 1,
    pin_memory: bool = True,
    seed: int = 1337,
):
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create directory for saving model and logs
    save_path = Path(train_dir) / name
    save_path.mkdir(parents=True, exist_ok=True)

    # Determine the device to use
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if device == "cuda":
        # Determine cuda device with most available memory
        device_with_most_available_memory = cuda_devices_sorted_by_free_mem()[-1]
        device = f"cuda:{device_with_most_available_memory}"

    # Datasets without per-atom offsets etc.
    train = Dataset.from_file(train_file)
    val = Dataset.from_file(val_file)
    train.to_units("eV")
    val.to_units("eV")

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_dicts,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_dicts,
        pin_memory=pin_memory,
    )

    model_kwargs = {}

    if fine_tune:
        logger.info("Fine-tuning model")
        # TODO make more general load model
        # load_model(path: str, params=None, model_type=None, **kwargs)
        # MACE is NffScaleMACE.load_foundations("medium", map_location="cpu", default_dtype="float32")
        # need to fix for MACE
        # load_foundations
        # model_path is empty to load foundational models
        model = load_model(model_path, model_type=model_type, map_location=device, device=device)
        # model = CHGNetNFF.load(device=device)

        # Optionally fix the weights of some layers
        if allow_grad == "ALL":
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
        elif allow_grad == "LAST":
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
        # Load model params and save a copy
        logger.info("Training model from scratch")
        try:
            with open(model_params_path, "r", encoding="utf-8") as f:
                model_params = yaml.safe_load(f)
            logger.info("Loaded model params from %s", model_params_path)
        except IOError:
            logger.error("Model params file %s not found!", model_params_path)

        with open(save_path / "model_params.yaml", "w", encoding="utf-8") as f:
            yaml.dump(
                model_params,
                f,
                default_flow_style=None,
                allow_unicode=True,
                sort_keys=False,
            )
        logger.info("Saved model params to save path %s", save_path)

        if "NffScaleMACE" in model_type:
            model_params = update_mace_init_params(train, val, train_loader, model_params, logger=logger)
        model_kwargs.update({"training": True, "compute_force": True})  # needs to be true for training
        model = get_model(model_params, model_type=model_type)

    model.to(device)
    model.float()

    logger.info(
        "Num trainable parameters: %s",
        sum([p.numel() for p in model.parameters() if p.requires_grad]),
    )

    train.to_units(model.units)
    val.to_units(model.units)

    # define loss
    loss_fn = loss.build_general_loss(
        loss_coef=dict(zip(targets, loss_weights)),
        operation=LOSS_OPERATIONS[criterion],
    )

    optimizer = torch.optim.Adam(  # TODO Adam or AdamW
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # monitor energy and force MAE
    train_metrics = [
        metrics.MeanAbsoluteError("energy"),
        metrics.MeanAbsoluteError("energy_grad"),
    ]

    train_hooks = [
        hooks.WarmRestartHook(
            T0=max_num_epochs,
            Tmult=1,
            min_lr=min_lr,
            lr_factor=lr,
            optimizer=optimizer,
        ),
        hooks.MaxEpochHook(max_num_epochs),
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
            patience=patience,
            factor=lr,
            min_lr=min_lr,
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
        model_kwargs=model_kwargs,
    )

    T.train(device=device, n_epochs=max_num_epochs)

    logger.info("Model saved in %s", save_path)


if __name__ == "__main__":
    args = build_default_arg_parser().parse_args()
    main(
        name=args.name,
        model_type=args.model_type,
        model_params_path=args.model_params_path,
        model_path=args.model_path,
        train_dir=args.train_dir,
        train_file=args.train_file,
        val_file=args.val_file,
        fine_tune=args.fine_tune,
        allow_grad=args.allow_grad,
        targets=args.targets,
        loss_weights=args.loss_weights,
        criterion=args.criterion,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        max_num_epochs=args.max_num_epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
    )
