import os
import pathlib

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler


from nff.data import Dataset, split_train_validation_test, collate_dicts
from nff.train import Trainer, get_model, loss, hooks, metrics, evaluate


def test_excited_training(device, tmpdir):
    # Loss function
    # Let's make a loss function for the model. We'll use three diabatic states,
    # so that the model outputs the six quantities `d_{ij}` for `i >= j`, and `0 <= i, j <= 2`.
    # That means the model outputs `d_00`, `d_01`, etc. The model will then produce three adiabatic energies,
    # `energy_{i}`. Last, the model will also run a backwards pass to produce the gradients `energy_{i}_grad`.
    # The loss function can penalize errors in the adiabatic energies and forces, adiabatic gaps,
    # and non-adiabatic couplings (NACV). It can also penalize errors in the `d_{i}{i}`,
    # provided that a set of reference geometries with known `d_{i}{i}` is in the dataset.

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

    # We see that each key is a different type of loss (e.g. `mse` for mean-squared-error),
    # and each value is a list of sub-dictionaries. Each sub-dictionary contains information about the quantities
    # being penalized, through `params`, and their relative weights, through `coef`.
    # For example, `d_11` is through with an MSE loss with weight 0.01. Some notes:
    #
    # - The NACV gets its own special loss type, called `nacv`.
    # This is because it must correct the phase of the predicted NACV to minimize the prediction error.
    # This accounts for random sign changes in the ground truth NACV.
    #
    # - Energy gaps in the dataset are denoted `energy_{i}_energy_{j}_delta`, where i > j.
    #
    # - Force NACVS in the dataset are denoted `force_nacv_{i}_energy_{j}_delta`, where i > j.
    #
    # Now we can supply the loss dictionary to `loss.build_multi_loss`, and we have our loss funnction:

    loss_fn = loss.build_multi_loss(loss_dict)

    # ## Making the model
    #
    # Now let's make the model. To do this we can use `get_model`, together with the model parameters and model type.
    #
    # Our model type is called `PainnDiabat`, because it's PaiNN with diabatic states. We'll also have to supply the keys below, which include the regular hyperparameters (activation function, feature dimension, etc.), as well as the keys of the diabatic energies. We'll also specify `add_nacv=True`, which means the NACV will get computed when the model is called, and hence can be used in the loss function:
    #

    # Only provide the 6 unique diabatic keys
    # It doesn't matter whether you use upper- or lower-triangular

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

    # ## Optimizer, metrics, and hooks
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=1e-4)

    train_metrics = [
        metrics.MeanAbsoluteError("energy_0"),
        metrics.MeanAbsoluteError("energy_1"),
        metrics.MeanAbsoluteError("energy_0_grad"),
        metrics.MeanAbsoluteError("energy_1_grad"),
        metrics.MeanAbsoluteError("energy_1_energy_0_delta"),
    ]

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

    # ## Dataset
    dset = Dataset.from_file(os.path.join(pathlib.Path(__file__).parent.absolute(), "data/azo_diabat.pth.tar"))
    train, val, test = split_train_validation_test(dset, val_size=0.1, test_size=0.1)

    batch_size = 20
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_dicts, sampler=RandomSampler(train))
    val_loader = DataLoader(val, batch_size=batch_size, collate_fn=collate_dicts)
    test_loader = DataLoader(test, batch_size=batch_size, collate_fn=collate_dicts)

    # ## Training

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

    # ## Evaluating

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
