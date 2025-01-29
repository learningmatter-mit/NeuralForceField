import os
import pathlib

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from nff.data import Dataset, collate_dicts, split_train_validation_test
from nff.train import Trainer, evaluate, get_model, hooks, loss, metrics


def test_training(device, tmpdir):
    # data set
    OUTDIR = tmpdir
    dataset = Dataset.from_file(os.path.join(pathlib.Path(__file__).parent.absolute(), "data", "dataset.pth.tar"))
    train, val, test = split_train_validation_test(dataset, val_size=0.2, test_size=0.2)
    train_loader = DataLoader(train, batch_size=50, collate_fn=collate_dicts)
    val_loader = DataLoader(val, batch_size=50, collate_fn=collate_dicts)
    test_loader = DataLoader(test, batch_size=50, collate_fn=collate_dicts)

    # define model
    params = {
        "n_atom_basis": 256,
        "n_filters": 256,
        "n_gaussians": 32,
        "n_convolutions": 4,
        "cutoff": 5.0,
        "trainable_gauss": True,
        "dropout_rate": 0.2,
    }
    model = get_model(params)

    # define training
    loss_fn = loss.build_mse_loss(loss_coef={"energy": 0.01, "energy_grad": 1})
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=3e-4)
    train_metrics = [metrics.MeanAbsoluteError("energy"), metrics.MeanAbsoluteError("energy_grad")]

    # output
    train_hooks = [
        hooks.MaxEpochHook(7),
        hooks.CSVHook(
            OUTDIR,
            metrics=train_metrics,
        ),
        hooks.PrintingHook(OUTDIR, metrics=train_metrics, separator=" | ", time_strf="%M:%S"),
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer, patience=30, factor=0.5, min_lr=1e-7, window_length=1, stop_after_min=True
        ),
    ]

    # train
    T = Trainer(
        model_path=OUTDIR,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        checkpoint_interval=1,
        hooks=train_hooks,
    )
    T.train(device=device, n_epochs=7)

    # evaluation
    results, targets, val_loss = evaluate(T.get_best_model(), test_loader, loss_fn, device=device)
    for key in ["energy_grad", "energy"]:
        pred = torch.stack(results[key], dim=0).view(-1).detach().cpu().numpy()
        targ = torch.stack(targets[key], dim=0).view(-1).detach().cpu().numpy()
        mae = abs(pred - targ).mean()
        assert mae < 10.0
