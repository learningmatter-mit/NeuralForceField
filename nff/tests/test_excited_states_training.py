import os
import pathlib

import pytest
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler

from nff.data import Dataset, collate_dicts, split_train_validation_test
from nff.train import Trainer, evaluate, get_model, hooks, loss, metrics


@pytest.mark.skip("still taking too long, disable for now")
def test_excited_training(device, tmpdir):
    # define loss
    loss_dict = {
        "mse": [
            {"coef": 0.01, "params": {"key": "d_00"}},
            {"coef": 0.01, "params": {"key": "d_11"}},
            {"coef": 0.01, "params": {"key": "d_22"}},
            {"coef": 0.2, "params": {"key": "energy_0"}},
            {"coef": 1, "params": {"key": "energy_0_grad"}},
            {"coef": 0.1, "params": {"key": "energy_1"}},
            {"coef": 1, "params": {"key": "energy_1_grad"}},
            {"coef": 0.5, "params": {"key": "energy_1_energy_0_delta"}},
        ],
        "nacv": [{"coef": 1, "params": {"abs": False, "key": "force_nacv_10", "max": False}}],
    }
    loss_fn = loss.build_multi_loss(loss_dict)

    # define model
    diabat_keys = [["d_00", "d_01", "d_02"], ["d_01", "d_11", "d_12"], ["d_02", "d_12", "d_22"]]
    modelparams = {
        "feat_dim": 128,
        "activation": "swish",
        "n_rbf": 20,
        "cutoff": 5.0,
        "num_conv": 3,
        "output_keys": ["energy_0", "energy_1"],
        "grad_keys": ["energy_0_grad", "energy_1_grad"],
        "diabat_keys": diabat_keys,
        "add_nacv": True,
    }
    model = get_model(modelparams, model_type="PainnDiabat")

    # define training
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=1e-4)
    train_metrics = [
        metrics.MeanAbsoluteError("energy_0"),
        metrics.MeanAbsoluteError("energy_1"),
        metrics.MeanAbsoluteError("energy_0_grad"),
        metrics.MeanAbsoluteError("energy_1_grad"),
        metrics.MeanAbsoluteError("energy_1_energy_0_delta"),
    ]

    # output
    outdir = tmpdir
    train_hooks = [
        hooks.CSVHook(
            outdir,
            metrics=train_metrics,
        ),
        hooks.PrintingHook(outdir, metrics=train_metrics, separator=" | ", time_strf="%M:%S"),
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer,
            # patience in the original paper
            patience=50,
            factor=0.5,
            min_lr=1e-7,
            window_length=1,
            stop_after_min=True,
        ),
    ]

    # data set
    dset = Dataset.from_file(os.path.join(pathlib.Path(__file__).parent.absolute(), "data/azo_diabat.pth.tar"))
    train, val, test = split_train_validation_test(dset, val_size=0.1, test_size=0.1)
    batch_size = 20
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_dicts, sampler=RandomSampler(train))
    val_loader = DataLoader(val, batch_size=batch_size, collate_fn=collate_dicts)
    test_loader = DataLoader(test, batch_size=batch_size, collate_fn=collate_dicts)

    # train
    T = Trainer(
        model_path=outdir,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        checkpoint_interval=1,
        hooks=train_hooks,
        mini_batches=1,
    )
    T.train(device=device, n_epochs=10)

    # evaluation
    def correct_nacv(results, targets, key):
        num_atoms = targets["num_atoms"]
        if not isinstance(num_atoms, list):
            num_atoms = num_atoms.tolist()
        pred = torch.split(torch.cat(results[key]), num_atoms)
        targ = torch.split(torch.cat(targets[key]), num_atoms)

        real_pred = []

        for p, t in zip(pred, targ):
            sub_err = (p - t).abs().mean()
            add_err = (p + t).abs().mean()
            sign = 1 if sub_err < add_err else -1
            real_pred.append(sign * p)

        return real_pred

    results, targets, test_loss = evaluate(
        T.get_best_model(), test_loader, loss_fn=lambda x, y: torch.Tensor([0]), device=device
    )
    real_nacv = correct_nacv(results, targets, "force_nacv_10")
    results["force_nacv_10"] = real_nacv

    en_keys = ["energy_0", "energy_1", "energy_1_energy_0_delta"]
    grad_keys = ["energy_0_grad", "energy_1_grad"]

    for key in [*en_keys, *grad_keys, "force_nacv_10"]:
        pred = results[key]
        targ = targets[key]
        targ_dim = len(targets["energy_0"][0].shape)
        fn = torch.stack if targ_dim == 0 else torch.cat
        pred = torch.cat(pred).reshape(-1)
        targ = fn(targ).reshape(-1)
        assert abs(pred - targ).mean() < 12.0
