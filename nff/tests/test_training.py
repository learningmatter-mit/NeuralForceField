import os
import pathlib

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from nff.data import Dataset, collate_dicts, split_train_validation_test, to_tensor
from nff.train import Trainer, evaluate, get_model, hooks, loss, metrics


def test_training(device, tmpdir):
    OUTDIR = tmpdir
    dataset = Dataset.from_file(os.path.join(pathlib.Path(__file__).parent.absolute(), "data", "dataset.pth.tar"))
    train, val, test = split_train_validation_test(dataset, val_size=0.2, test_size=0.2)

    train_loader = DataLoader(train, batch_size=50, collate_fn=collate_dicts)
    val_loader = DataLoader(val, batch_size=50, collate_fn=collate_dicts)
    test_loader = DataLoader(test, batch_size=50, collate_fn=collate_dicts)

    # `nff` is based on SchNet. It parameterizes interatomic interactions in molecules and materials through a
    # series of convolution layers with continuous filters. Here, we are going to create a simple model using the
    # hyperparameters given on `params`:
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

    # ## Creating a trainer

    # To train our model with the data provided, we have to create a loss function.
    # The easiest way to do that is through the `build_mse_loss` builder.
    # Its argument `rho` is a parameter that will multiply the mean square error (MSE) of the force components
    # before summing it with the MSE of the energy.
    loss_fn = loss.build_mse_loss(loss_coef={"energy": 0.01, "energy_grad": 1})

    # We should also select an optimizer for our recently created model:
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=3e-4)

    # ### Metrics and hooks
    # Metrics and hooks allow the customization of the training process.
    # Instead of tweaking directly the code or having to resort to countless flags,
    # we can create submodules (or add-ons) to monitor the progress of the training or customize it.
    # If we want to monitor the progress of our training, say by looking at the mean absolute error (MAE) of
    # energies and forces, we can simply create metrics to observe them:

    train_metrics = [metrics.MeanAbsoluteError("energy"), metrics.MeanAbsoluteError("energy_grad")]

    # Furthermore, if we want to customize how our training procedure is done, we can use hooks which can interrupt
    # or change the train automatically.
    #
    # In our case, we are adding hooks to:
    # * Stop the training procedure after 100 epochs;
    # * Log the training on a machine-readable CSV file under the directory `./sandbox`;
    # * Print the progress on the screen with custom formatting; and
    # * Set up a scheduler for the learning rate.
    # the max epochs are massively reduced to be suitable for a CI pipeline

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

    # ### Trainer wrapper
    # A `Trainer` in the `nff` package is a wrapper to train a model. It automatically creates checkpoints,
    # as well as trains and validates a given model.
    # It also allows further training by loading checkpoints from existing paths,
    # making the training procedure more flexible.
    # Its functionality can be extended by the hooks we created above.
    # To create a trainer, we have to execute the following command:
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

    # Now we can finally train the model using the method `train` from the `Trainer`:
    # the number of epochs is massively reduced to be suitable for a CI pipeline
    T.train(device=device, n_epochs=7)

    # ## Evaluating the model on the test set
    # Now we have a brand new model trained and validated.
    # We can use the best model from this training to evaluate its performance on the test set.
    # `results` contains the predictions of properties for the whole test dataset.
    # `targets` contains the ground truth for such data.
    # `test_loss` is the loss, calculated with the same function used during the training part
    results, targets, val_loss = evaluate(T.get_best_model(), test_loader, loss_fn, device=device)

    to_tensor(results["energy"], stack=True).shape

    # Finally, we could plot our results to observe how well is our model performing
    # here we are just checking the MAE to ensure the pipeline works

    keys = "energy_grad", "energy"
    for key in keys:
        pred = torch.stack(results[key], dim=0).view(-1).detach().cpu().numpy()
        targ = torch.stack(targets[key], dim=0).view(-1).detach().cpu().numpy()
        mae = abs(pred - targ).mean()
        assert mae < 10.0
