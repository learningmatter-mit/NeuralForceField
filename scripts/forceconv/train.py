
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from nff.data import Dataset, split_train_validation_test, collate_dicts, to_tensor
from nff.train import Trainer, get_trainer, get_model, load_model, loss, hooks, metrics, evaluate
from MD17data import *

def train(params, suggestion, model, n_epochs, angle=False):
    trainparam = suggestion.assignments
    
    # get data 
    data = get_MD17data(params['data'])
    dataset = pack_MD17data(data, 10000)

    if angle:
        dataset.generate_angle_list()
        dataset.generate_kj_ji()


    train, val, test = split_train_validation_test(dataset, val_size=0.05, test_size=0.85)
    train_loader = DataLoader(train, batch_size=trainparam['batch_size'], collate_fn=collate_dicts)
    val_loader = DataLoader(val, batch_size=trainparam['batch_size'], collate_fn=collate_dicts)
    test_loader = DataLoader(test, batch_size=trainparam['batch_size'], collate_fn=collate_dicts)

    model = model(**trainparam)
    loss_fn = loss.build_mse_loss(loss_coef={'energy_grad': 1})
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=trainparam['lr'])
    train_metrics = [
        metrics.MeanAbsoluteError('energy_grad')
    ]

    DEVICE = params['device']
    OUTDIR = '{}/{}/sandbox'.format(params['logdir'], suggestion.id)

    print(OUTDIR)

    if os.path.exists(OUTDIR):
        newpath = os.path.join(os.path.dirname(OUTDIR), 'backup')
        if os.path.exists(newpath):
            shutil.rmtree(newpath)
        shutil.move(OUTDIR, newpath)

    train_hooks = [
        hooks.MaxEpochHook(n_epochs),
        hooks.CSVHook(
            OUTDIR,
            metrics=train_metrics,
        ),
        hooks.PrintingHook(
            OUTDIR,
            metrics=train_metrics,
            separator = ' | ',
            time_strf='%M:%S'
        ),
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer,
            patience=20,
            factor=0.5,
            min_lr=1e-7,
            window_length=1,
            stop_after_min=True
        )
    ]

    T = Trainer(
        model_path=OUTDIR,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        checkpoint_interval=1,
        hooks=train_hooks
    )

    print("Start training ")
    T.train(device=DEVICE, n_epochs=n_epochs)

    # evaluate model
    # get new data  
    # data = get_MD17data(params['data'])
    # dataset = pack_MD17data(data, 1000)
    # test_loader =  DataLoader(dataset, batch_size=trainparam['batch_size'], collate_fn=collate_dicts)

    results, targets, val_loss = evaluate(model, test_loader, loss_fn, device=DEVICE)
    key = 'energy_grad'
    pred = torch.stack(results[key][:-1], dim=0).view(-1).detach().cpu().numpy()
    targ = torch.stack(targets[key][:-1], dim=0).view(-1).detach().cpu().numpy()
    mae = np.abs(pred - targ).mean()

    print("Test loss for {}: {} kcal/(mol A)".format(params['data'], mae))
    
    return mae 