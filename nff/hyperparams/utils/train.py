from nff.train import loss, metrics, hooks, Trainer
from torch.optim import Adam

def get_loss(loss_name, loss_coef):
    loss_build_name = "build_{}_loss".format(loss_name)
    loss_builder = getattr(loss, loss_build_name)
    loss_fn = loss_builder(loss_coef=loss_coef)

    return loss_fn


def make_hooks(max_epochs, train_metrics, optimizer, model_folder, min_lr,
               patience, factor):

    train_hooks = [
        hooks.MaxEpochHook(max_epochs),
        hooks.CSVHook(
            model_folder,
            metrics=train_metrics,
        ),
        hooks.PrintingHook(
            model_folder,
            metrics=train_metrics,
            separator=' | ',
            time_strf='%M:%S'
        ),
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer,
            patience=patience,
            factor=factor,
            min_lr=min_lr,
            window_length=1,
            stop_after_min=True
        )
    ]

    return train_hooks


def make_metrics(metric_dics):
    """
    Example:
        [{"target": "bind", "metric": "TruePositives",
        "target": "energy_grad", "metric": "MAE"}]
    """

    train_metrics = []

    for dic in metric_dics:
        target = dic["target"]
        metric_func = getattr(metrics, dic["metric"])
        metric = metric_func(target)
        train_metrics.append(metric)

    return train_metrics


def make_trainer(model,
                 train_loader,
                 val_loader,
                 model_folder,
                 loss_name,
                 loss_coef,
                 metric_dics,
                 max_epochs,
                 min_lr=1e-7,
                 patience=30,
                 factor=0.5):

    loss_fn = get_loss(loss_name=loss_name,
                       loss_coef=loss_coef)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=1e-5)
    train_metrics = make_metrics(metric_dics)

    train_hooks = make_hooks(max_epochs=max_epochs,
                             train_metrics=train_metrics,
                             optimizer=optimizer,
                             model_folder=model_folder,
                             min_lr=min_lr,
                             patience=patience,
                             factor=factor)
    T = Trainer(
        model_path=model_folder,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        checkpoint_interval=1,
        hooks=train_hooks
    )
    return T
