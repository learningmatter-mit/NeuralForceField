import argparse
import logging
from pathlib import Path
from typing import Iterable, Literal, Union

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from nff.data import Dataset, collate_dicts
from nff.data.dataset import to_tensor
from nff.io.mace import update_mace_init_params
from nff.nn.models.mace import reduce_foundations
from nff.train import Trainer, get_layer_freezer, get_model, hooks, load_model, loss, metrics
from nff.train.loss import mae_operation, mse_operation
from nff.utils.cuda import cuda_devices_sorted_by_free_mem

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
    parser.add_argument(
        "--custom_layers",
        nargs="+",
        type=str,
        default=[],
        help="Which layers to unfreeze for fine-tuning",
    )
    parser.add_argument(
        "--freeze_pooling",
        help="Whether to freeze pooling layers for fine-tuning",
        action="store_true",
    )
    parser.add_argument(
        "--unfreeze_embeddings",
        help="Whether to unfreeze embeddings for fine-tuning",
        action="store_true",
    )
    parser.add_argument(
        "--unfreeze_conv_layers", help="Number of convolutional layers to unfreeze for fine-tuning", type=int, default=1
    )
    parser.add_argument(
        "--unfreeze_interactions", help="Whether to unfreeze all MACE interactions for fine-tuning", action="store_true"
    )
    parser.add_argument(
        "--trim_embeddings",
        help="Whether to reduce the size of MACE foundational model by resizing the embedding layers",
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
        "--lr_decay",
        help="Factor to reduce learning rate by at each step. lr = lr * lr_decay",
        type=float,
        default=0.5,
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


def main(
    name: str,
    model_type: str,
    model_params_path: Union[str, Path],
    model_path: Union[str, Path],
    train_dir: Union[str, Path],
    train_file: Union[str, Path],
    val_file: Union[str, Path],
    fine_tune: bool = False,
    custom_layers: Iterable[str] = [],
    freeze_pooling: bool = False,
    unfreeze_embeddings: bool = False,
    unfreeze_conv_layers: int = 1,
    unfreeze_interactions: bool = False,
    trim_embeddings: bool = False,
    targets: Iterable[str] = ["energy", "energy_grad"],
    loss_weights: Iterable[float] = [0.05, 1.0],
    criterion: Literal["MSE", "MAE"] = "MSE",
    batch_size: int = 16,
    lr: float = 1e-3,
    min_lr: float = 1e-6,
    max_num_epochs: int = 200,
    patience: int = 25,
    lr_decay: float = 0.5,
    weight_decay: float = 0.0,
    num_workers: int = 1,
    pin_memory: bool = True,
    seed: int = 1337,
):
    """Train a neural network model.

    Args:
        name (str): Model name
        model_type (str): Type of model
        model_params_path (Union[str, Path]): Path to model parameters
        model_path (Union[str, Path]): Path to a trained model
        train_dir (Union[str, Path]): Model training directory
        train_file (Union[str, Path]): Training set pth.tar file
        val_file (Union[str, Path]): Validation set pth.tar file
        fine_tune (bool, optional): Whether to fine tune an existing model. Defaults to False.
        custom_layers (Iterable[str], optional): Named modules to unfreeze for finetuning. Defaults to [].
        freeze_pooling (bool, optional): Whether to freeze pooling layers for fine-tuning. Defaults to False.
        unfreeze_embeddings (bool, optional): Whether to unfreeze embeddings for fine-tuning. Defaults to False.
        unfreeze_conv_layers (int, optional): Number of convolutional layers to unfreeze for fine-tuning. Defaults to 1.
        unfreeze_interactions (bool, optional): Whether to unfreeze all MACE interactions for fine-tuning. Defaults to False.
        trim_embeddings (bool, optional): Whether to trim MACE embeddings. Defaults to False.
        targets (Iterable[str], optional): Model output. Defaults to ["energy", "energy_grad"].
        loss_weights (Iterable[float], optional): Relative weights of output targets. Defaults to [0.05, 1.0].
        criterion (Literal[&quot;MSE&quot;, &quot;MAE&quot;], optional): Loss function criterion. Defaults to "MSE".
        batch_size (int, optional): Batch size. Defaults to 16.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        min_lr (float, optional): Minimum LR. Defaults to 1e-6.
        max_num_epochs (int, optional): Max number training epochs. Defaults to 200.
        patience (int, optional): LR patience. Defaults to 25.
        lr_decay (float, optional): LR decay rate. Defaults to 0.5.
        weight_decay (float, optional): Weight decay for optimizer. Defaults to 0.0.
        num_workers (int, optional): Number of workers for data loader. Defaults to 1.
        pin_memory (bool, optional): Whether to pin memory for data loader. Defaults to True.
        seed (int, optional): Random seed. Defaults to 1337.
    """
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

    model_kwargs = {"training": True, "compute_force": True} if "NffScaleMACE" in model_type else {}

    if fine_tune:
        logger.info("Fine-tuning model")
        model = load_model(model_path, model_type=model_type, map_location=device, device=device)
        if "NffScaleMACE" in model_type and trim_embeddings:
            atomic_numbers = to_tensor(train.props["nxyz"], stack=True)[:, 0].unique().to(int).tolist()
            logger.info("Trimming embeddings with MACE model and atomic numbers %s", atomic_numbers)
            model = reduce_foundations(model, atomic_numbers, load_readout=True)
        model_freezer = get_layer_freezer(model_type)
        if unfreeze_conv_layers > 0:
            model_freezer.model_tl(
                model,
                custom_layers=custom_layers,
                freeze_interactions=not unfreeze_interactions,  # freeze MACE all interactions (all conv parameters, not
                # just the linear layers)
                freeze_pooling=freeze_pooling,
                unfreeze_conv_layers=unfreeze_conv_layers,
                unfreeze_embeddings=unfreeze_embeddings,
            )
        else:
            model_freezer.model_tl(
                model,
                custom_layers=custom_layers,
                freeze_interactions=not unfreeze_interactions,
                freeze_pooling=freeze_pooling,
                unfreeze_embeddings=unfreeze_embeddings,
            )
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
        model = get_model(model_params, model_type=model_type)

    model.to(device)

    logger.info("Total number of parameters: %s", sum(p.numel() for p in model.parameters()))

    logger.info(
        "Num trainable parameters: %s",
        sum([p.numel() for p in model.parameters() if p.requires_grad]),
    )

    model_units = model.units if hasattr(model, "units") else "kcal/mol"
    train.to_units(model_units)
    val.to_units(model_units)

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
            factor=lr_decay,
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
        custom_layers=args.custom_layers,
        freeze_pooling=args.freeze_pooling,
        unfreeze_embeddings=args.unfreeze_embeddings,
        unfreeze_conv_layers=args.unfreeze_conv_layers,
        unfreeze_interactions=args.unfreeze_interactions,
        trim_embeddings=args.trim_embeddings,
        targets=args.targets,
        loss_weights=args.loss_weights,
        criterion=args.criterion,
        batch_size=args.batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        max_num_epochs=args.max_num_epochs,
        patience=args.patience,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
    )
