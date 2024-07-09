"""
Train a CP3D model.
"""

import os
import argparse
import json
import sys
import copy
import pickle
import numpy as np
import shutil

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from nff.data import Dataset, split_train_validation_test, collate_dicts
from nff.data.loader import (ImbalancedDatasetSampler, BalancedFFSampler,
                             BalancedBatchedSpecies)
from nff.train import metrics, Trainer, load_model, get_model, loss, hooks
from nff.utils.confs import trim_confs
from nff.utils import fprint, tqdm_enum
from nff.train.transfer import painn_diabat_tl, painn_tl

# from nff.nn.models import Painn, PainnDiabat

import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

DEFAULTPARAMSFILE = 'job_info.json'
DEFAULT_METRICS = ["MeanAbsoluteError"]
DEFAULT_CUTOFF = 5.0


def init_parallel(node_rank,
                  gpu,
                  gpus,
                  world_size,
                  torch_par):
    """
    Initialize parallel training.
    Args:
        node_rank (int): rank of the current node
        gpu (int): local rank of the gpu on the current node
        gpus (int): number of gpus per node
        world_size (int): total number number of gpus altogether
        torch_par (bool): whether or not to use torch parallelization
            (alternative is just writing gradients to disk and loading
            them)
    """

    # global rank
    rank = node_rank * gpus + gpu

    if torch_par:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

    print("Hello from rank {}".format(rank))
    sys.stdout.flush()


def load_dset(path,
              max_confs,
              rank,
              needs_nbrs,
              needs_angles,
              cutoff):
    """
    Load a dataset and trim its conformers if requested.
    Args:
        path (str): path to the dataset
        max_confs (int): maximum number of conformers per
            species.
        rank (int): global rank of the current process
    Returns:
        dset (nff.data.Dataset): loaded dataset
    """

    dset = Dataset.from_file(path)
    base = (rank == 0)

    gen_nbrs = ((needs_nbrs or needs_angles)
                and "nbr_list" not in dset.props)
    gen_angles = (needs_angles and "angle_list"
                  not in dset.props)
    save = (gen_nbrs or gen_angles)
    gprint = fprint if base else lambda x: None

    if gen_nbrs:
        gprint(("Generating neighbor list with cutoff "
                "%.2f Angstroms..." % cutoff))
        dset.generate_neighbor_list(cutoff, undirected=False)
        gprint("Completed neighbor list generation!")

    if gen_angles:
        gprint(("Generating angle list and directed "
                "indices..."))
        dset.generate_angle_list()
        gprint("Completed angle list generation!")

    if max_confs is not None:
        # only track progress if this is the base process
        if base:
            enum_func = tqdm_enum
        else:
            enum_func = enumerate
        dset = trim_confs(dataset=dset,
                          num_confs=max_confs,
                          idx_dic=None,
                          enum_func=enum_func)

    if save:
        gprint("Saving dataset...")
        dset.save(path)
        gprint("Done saving dataset!")

    return dset


def get_gpu_splits(weight_path,
                   rank,
                   world_size,
                   params,
                   max_confs,
                   needs_nbrs,
                   needs_angles,
                   cutoff):
    """ 
    Check if there are already datasets in each parallel folder.
    If so, load and return those datasets instead of loading the whole
    thing in memory on every gpu and splitting afterwards.

    Args:
        weight_path (str): training folder
        rank (int): global rank of the current process
        world_size (int): total number number of gpus altogether
        params (dict): training/network parameters
        max_confs (int): maximum number of conformers per
            species.

    Returns:
        datasets (list): train, val, and test get_datasets if
            the datasets have already been split by GPU.
            None otherwise.

    """

    # get the parallel folders: weight_path / {0, 1, 2, ..., n_gpus}

    par_folders = [os.path.join(weight_path, folder) for
                   folder in os.listdir(weight_path) if
                   folder.isdigit()]

    # see if the data has already been split by gpu

    train_splits = ["train.pth.tar", "val.pth.tar", "test.pth.tar"]
    dset_name = "dataset.pth.tar"
    has_train_splits = all([name in os.listdir(folder) for name in train_splits
                            for folder in par_folders])
    has_dset = all([dset_name in os.listdir(folder) for folder in par_folders])
    has_splits = (has_train_splits or has_dset) and len(
        par_folders) >= world_size

    # if not, return None

    if not has_splits:
        return

    dat_path = os.path.join(weight_path, str(rank), "dataset.pth.tar")
    split_paths = [os.path.join(weight_path, str(rank), name + ".pth.tar")
                   for name in ["train", "val", "test"]]

    # if the train/val/test splits are already saved, then load them

    if all([os.path.isfile(path) for path in split_paths]):
        if max_confs is not None and (rank == 0):
            conf_str = "conformer" if max_confs == 1 else "conformers"
            fprint(("Reducing each species to have a maximum of "
                    f"{max_confs} {conf_str}..."))
        datasets = []
        for path in split_paths:
            dset = load_dset(path=path,
                             max_confs=max_confs,
                             rank=rank,
                             needs_nbrs=needs_nbrs,
                             needs_angles=needs_angles,
                             cutoff=cutoff)
            datasets.append(dset)

        return datasets

    # otherwise get the dataset, split it, and save it

    dataset = load_dset(dat_path, max_confs, rank)

    # split this sub-dataset into train/val/test

    train, val, test = split_train_validation_test(
        dataset,
        val_size=params['split'][0],
        test_size=params['split'][1]
    )

    datasets = [train, val, test]

    # save the splits to the training folder

    names = ['train', 'val', 'test']
    for d_set, name in zip(datasets, names):
        data_path = os.path.join(weight_path, str(rank),
                                 "{}.pth.tar".format(name))
        d_set.save(data_path)

    return datasets


def dic_to_tensor(dic):
    for key, val in dic.items():
        if isinstance(val, dict):
            dic[key] = dic_to_tensor(val)
        else:
            if isinstance(val, list) or isinstance(val, np.ndarray):
                if all([not isinstance(sub_val, str)
                        for sub_val in val]):
                    dic[key] = torch.tensor(val)
                else:
                    dic[key] = val
            else:
                dic[key] = val
    return dic


def load_ff_sampler(weight_path,
                    rank,
                    sampler_path):

    full_path = os.path.join(weight_path, str(rank), sampler_path)
    if full_path.endswith("json"):
        with open(full_path, "r") as f_open:
            balance_dict = dic_to_tensor(json.load(f_open))
    elif full_path.endswith("pickle"):
        with open(full_path, "rb") as f_open:
            balance_dict = dic_to_tensor(pickle.load(f_open))
    else:
        raise NotImplementedError

    sampler = BalancedFFSampler(balance_dict=balance_dict)

    return sampler


def get_sampler(sample_params,
                dataset,
                rank,
                world_size,
                central_data,
                torch_par,
                dset_name,
                weight_path,
                batch_size):
    """
    Get sampler for instantiating the DataLoader.
    Args:
        params (dict): training/network parameters
        dataset (nff.data.Dataset): dataset to load
        rank (int): global rank of the current process
        world_size (int): total number number of gpus altogether
        central_data (bool): whether the data is centralized in one
            location or divided into GPU folders.
        torch_par (bool): whether or not to use torch parallelization
            (alternative is just writing gradients to disk and loading
            them)
        dset_name (str): train, val or test
        weight_path (str): training folder
    Returns:
        sample: the Data sampler

    Example:
        1. Asking for a sampler for imbalanced data 
            params = {"sampler": {"name": "ImbalancedDataSampler",
                                  "target_name": "bind"}}

    """

    # See if any specific sampler is mentioned in `params`

    apply_to = sample_params.get("apply_to", ["train"])
    custom_sampler = (dset_name in apply_to)

    sampler = None
    batch_sampler = None

    if not central_data:
        if not custom_sampler:
            pass

        elif sample_params.get("name") in ["BalancedFFSampler",
                                           "BalancedBatchedSpecies"]:

            sampler_path = sample_params.get("sampler_paths",
                                             {}).get(dset_name)
            if not sampler_path:
                msg = ("BalancedFFSampler must have pre-set sample "
                       "probabilities when doing parallel training.")
                raise Exception(msg)

            balanced_sampler = load_ff_sampler(weight_path=weight_path,
                                               rank=rank,
                                               sampler_path=sampler_path)

            if sample_params["name"] == "BalancedBatchedSpecies":
                smiles_list = dataset.props["smiles"]
                min_geoms = sample_params.get("min_geoms")
                batch_sampler = BalancedBatchedSpecies(
                    base_sampler=balanced_sampler,
                    smiles_list=smiles_list,
                    batch_size=batch_size,
                    min_geoms=min_geoms)
            else:
                sampler = balanced_sampler

        elif sample_params.get("name") == "ImbalancedDatasetSampler":
            target_name = sample_params["target_name"]
            sampler = ImbalancedDatasetSampler(target_name=target_name,
                                               props=dataset.props)
        elif sample_params.get("name") is not None:
            raise NotImplementedError((f"Sampler {sample_params['name']} not "
                                       "yet implemented"))

    elif central_data:
        # if not centralized, can't yet use a custom sampler
        if "name" in sample_params:
            raise NotImplementedError(("Cannot yet use a "
                                       "custom sampler without "
                                       "pre-splitting the data"))
        elif torch_par:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            raise Exception(("Must de-centralize data if you want to "
                             "avoid torch parallelization"))

    return sampler, batch_sampler


def make_loader(dataset,
                sample_params,
                batch_size,
                world_size,
                rank,
                central_data,
                torch_par,
                dset_name,
                weight_path):
    """
    Make distributed loader for a dataset.
    Args:
        dataset (nff.data.Dataset): dataset to load
        batch_size (int): per-gpu batch size (so the
            effective batch size is batch_size x world_size)
        world_size (int): total number number of gpus altogether
        rank (int): global rank of the current process
        central_data (bool): whether the data is centralized in one
            location or divided into GPU folders.
        torch_par (bool): whether or not to use torch parallelization
            (alternative is just writing gradients to disk and loading
            them)
        dset_name (str): train, val or test
        weight_path (str): training folder

    Returns:
        loader (torch.utils.data.DatalLoader): data loader
            for the dataset.

    """

    # get the dictionary with information about the sampler to be used
    # in the loader

    sampler, batch_sampler = get_sampler(sample_params=sample_params,
                                         dataset=dataset,
                                         rank=rank,
                                         world_size=world_size,
                                         central_data=central_data,
                                         dset_name=dset_name,
                                         torch_par=torch_par,
                                         weight_path=weight_path,
                                         batch_size=batch_size)
    if sampler:
        kwargs = {"sampler": sampler,
                  "batch_size": batch_size}
    elif batch_sampler:
        kwargs = {"batch_sampler": batch_sampler}
    else:
        kwargs = {"batch_size": batch_size}
    # make the loader with the custom collate function,
    # and shuffle=False, and any sampler information

    loader = DataLoader(
        dataset=dataset,
        collate_fn=collate_dicts,
        **kwargs)

    return loader


def make_all_loaders(weight_path,
                     rank,
                     world_size,
                     params,
                     base,
                     batch_size,
                     node_rank,
                     gpu,
                     gpus,
                     log_train,
                     needs_nbrs,
                     needs_angles,
                     cutoff):
    """
    Get train, val, and test data loaders.
    Args:
        weight_path (str): training folder
        rank (int): global rank of the current process
        world_size (int): total number number of gpus altogether
        params (dict): training/network parameters
        base (bool): whether this is the base process
        geoms (QuerySet): Geom objects to use when creating the dataset
        batch_size (int): per-gpu batch size
        node_rank (int): rank of the node
        gpu (int): local rank of the current gpu
        gpus (int): number of gpus per node
        log_train (Callable): train logger

    Returns:
        loaders (list): list of test, train and val loaders


    """

    # see if the data has been pre-split based on the number
    # of gpus

    max_confs = params.get("max_confs")
    gpu_splits = get_gpu_splits(weight_path=weight_path,
                                rank=rank,
                                world_size=world_size,
                                params=params,
                                max_confs=max_confs,
                                needs_nbrs=needs_nbrs,
                                needs_angles=needs_angles,
                                cutoff=cutoff)

    # if not, and if this is the base GPU, we need to either
    # load the dataset or make the dataset and save it to the
    # main folder

    if gpu_splits is None:

        # data is centralized
        central_data = True

        # If we're on gpu 0 and node 0, make the dataset.
        # Otherwise, load the dataset after calling
        #  `init_parallel`, which will wait for the process on
        # the base gpu to initialize.

        if base:
            train, val, test = dsets_from_folder(weight_path=weight_path,
                                                 max_confs=max_confs,
                                                 rank=rank,
                                                 needs_nbrs=needs_nbrs,
                                                 needs_angles=needs_angles,
                                                 cutoff=cutoff)
    else:

        central_data = False
        train, val, test = gpu_splits

    # initalize parallelizaiton

    torch_par = params.get("torch_par", True)
    init_parallel(node_rank=node_rank,
                  gpu=gpu,
                  gpus=gpus,
                  world_size=world_size,
                  torch_par=torch_par)

    # if the data hasn't been pre-split, and if this isn't the base GPU,
    # load the entire dataset from the main folder

    if gpu_splits is None and not base:
        train, val, test = dsets_from_folder(weight_path=weight_path,
                                             max_confs=max_confs,
                                             rank=rank)

    # record dataset stats

    d_set_size = len(train) + len(val) + len(test)
    log_train('{} data points'.format(d_set_size))

    # create data loaders

    dset_names = ["train", "val", "test"]
    dsets = [train, val, test]

    train_len = len(dsets[0])
    len_path = os.path.join(weight_path, str(rank), "train_len")
    with open(len_path, "w") as f:
        f.write(str(train_len))

    all_sample_params = params.get("sampler", [{}])
    if isinstance(all_sample_params, dict):
        all_sample_params = [all_sample_params]
    loader_sets = []
    for sample_params in all_sample_params:
        loaders = []
        for i, d_set in enumerate(dsets):
            loader = make_loader(dataset=d_set,
                                 sample_params=sample_params,
                                 batch_size=batch_size,
                                 world_size=world_size,
                                 rank=rank,
                                 central_data=central_data,
                                 torch_par=torch_par,
                                 dset_name=dset_names[i],
                                 weight_path=weight_path)

            loaders.append(loader)
        loader_sets.append(loaders)

    return loader_sets


def dsets_from_folder(weight_path,
                      max_confs,
                      rank,
                      needs_nbrs,
                      needs_angles,
                      cutoff):
    """
    Load train, val, and test datasets from the main folder.
    Args:
        weight_path (str): training folder
        rank (int): global rank of the current process
    Returns:
        datasets (list): train, val, and test datasets
        max_confs (int): maximum number of conformers per
            species.

    """

    names = ['train', 'val', 'test']
    datasets = []
    for name in names:
        data_path = os.path.join(weight_path, "{}.pth.tar".format(name))
        dset = load_dset(path=path,
                         max_confs=max_confs,
                         rank=rank,
                         needs_nbrs=needs_nbrs,
                         needs_angles=needs_angles,
                         cutoff=cutoff)
        datasets.append(dset)

    return datasets


def nff_to_splits(dataset, params, weight_path):
    """
    Get the train/val/test splits from an NFF dataset just
    created.
    Args:
        dataset: nff.data.dataset: NFF dataset
        params (dict):   training/network parameters
        weight_path (str): training folder
    Returns:
        datasets (list): train, val, and test datasets

    """

    # creating dataloader for training
    train, val, test = split_train_validation_test(
        dataset,
        val_size=params['split'][0],
        test_size=params['split'][1]
    )

    if not os.path.isdir(weight_path):
        os.makedirs(weight_path)

    datasets = [train, val, test]

    # save the splits to the training folder

    names = ['train', 'val', 'test']
    for d_set, name in zip(datasets, names):
        data_path = os.path.join(weight_path, "{}.pth.tar".format(name))
        d_set.save(data_path)

    return datasets


def load_transfer(params,
                  weight_path,
                  transfer_id,
                  log_train):

    direc = os.path.dirname(weight_path)
    transfer_path = os.path.join(direc, str(transfer_id))
    log_train(f"Loading model from {transfer_path}")

    model_type = params["model_type"]
    model = load_model(transfer_path,
                       params=params,
                       model_type=model_type)

    freeze_params = params.get("freeze_params")
    if not freeze_params:
        log_train("Not freezing any parameters")
        return model

    if model_type == 'PainnDiabat':
        painn_diabat_tl(model=model,
                        **freeze_params)
    elif model_type == 'Painn':
        painn_tl(model=model,
                 **freeze_params)
    else:
        raise NotImplementedError

    log_train(("Freezing model parameters using the "
               "following instructions:\n"
               f"{json.dumps(freeze_params, indent=4)}"))

    return model


def make_model(params,
               device,
               world_size,
               weight_path,
               log_train):
    """
    Create a model and wrap it in DistributedDataParallel.
    Args:
        params (dict):   training/network parameters
        device (int): local rank of the current gpu
        world_size (int): total number number of gpus altogether

    Returns:
        model (nn.parallel.DistributedDataParallel): wrapped
            moddel.

    """

    transfer_id = params.get("transferfrom")
    if transfer_id is not None:
        model = load_transfer(params=params,
                              weight_path=weight_path,
                              transfer_id=transfer_id,
                              log_train=log_train)

    else:
        model = get_model(params=params,
                          model_type=params.get("model_type", "SchNet"))

    if device != "cpu":
        torch.cuda.set_device(device)
    model.to(device)

    torch_par = params.get("torch_par", True)
    if torch_par:
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[device])
    return model


def is_base(gpu, node_rank):
    """
    Whether the current process is the base process.
    Args:
        gpu (int): local rank of the current gpu
        node_rank (int): rank of the node
    Returns:
        bool
    """

    return gpu == 0 and node_rank == 0


def get_nn_quants(all_params):
    """
    Get a variety of quantities for the neural network.
    Args:
        all_params (dict): all parameters in the job_info.json
            file
    Returns:
        model_name (str): id of the neural network
        params (dict): job params
        nn (neuralnet.models.NnPotential): neural network database object
        geoms (QuerySet): Geom objects to use when creating the dataset
        weight_path (str): training folder

    """

    params = {**all_params,
              **all_params.get('train_params', {}),
              **all_params.get('model_params', {}),
              **all_params.get('details', {})}
    model_name = params.get('model_name', params.get('nnid'))
    base_dir = params['weightpath']
    if not os.path.isdir(str(base_dir)):
        base_dir = params['mounted_weightpath']
    assert os.path.isdir(base_dir)
    weight_path = os.path.join(base_dir, str(model_name))
    if not os.path.isdir(weight_path):
        os.makedirs(weight_path)

    return model_name, params, weight_path


def init_quants(node_rank, gpu, gpus, world_size, params):
    """
    Initialize some parallelization quantities.
    Args:
        node_rank (int): rank of the node
        gpu (int): local rank of the current gpu
        gpus (int): number of gpus per node
        world_size (int): total number number of gpus altogether
        params (dict):   training/network parameters
    Returns:
        rank (int): global rank
        batch_size (int): per-gpu batch size
        base (bool): whether this is the base process
        log_train (Callable): train logger
    """

    rank = node_rank * gpus + gpu
    batch_size = int(params["batch_size"] / world_size)
    base = is_base(gpu=gpu, node_rank=node_rank)

    def log_train(msg):
        print('    TRAIN: ' + msg) if base else None
        sys.stdout.flush()

    # need same manual seed if not using torch parallelization
    # to sync up models
    if not params.get("torch_par", True):
        seed = params.get("seed", 0)
        torch.manual_seed(seed)
        log_train(f"Using seed {seed}")

    return rank, batch_size, base, log_train


def make_stats(trainer,
               test_loader,
               params,
               weight_path,
               device,
               val_loader,
               log_train):
    """
    Make test stats.
    Args:
        T (Trainer): model trainer
        test_loader (DataLoader): data loader for the test ste
        loss_fn (Callable): loss function
        params (dict):   training/network parameters
        weight_path (str): training folder
        global_rank (int): global rank of the current process
        base (bool): whether or not the current process is the
            base
        world_size (int): total number of gpus
        base_keys (list): list of properties predicted by the
            network
        grad_keys (list): properties whose gradients are also
            computed by the network.
    Returns:
        None
    """

    # old model
    old_model = copy.deepcopy(trainer._model.to("cpu"))

    # get best model and put into eval mode
    while True:
        try:
            model = load_model(weight_path,
                               params=params,
                               model_type=params["model_type"])
            break
        except (EOFError, FileNotFoundError, pickle.UnpicklingError):
            continue

    model.eval()
    trainer._model = model.to(device)

    # set the trainer validation loader to the test
    # loader so you can use its metrics to get the
    # performance on the test set

    trainer.validation_loader = test_loader

    # validate on test loader
    trainer.validate(device, test=True)

    # get the metric performance

    log_hook = [h for h in trainer.hooks if isinstance(
        h, hooks.PrintingHook)][0]
    final_stats = log_hook.aggregate(trainer, test=True)

    # param_path = os.path.join(weight_path, "params.json")

    # save the stats in the model directory and in the job
    # directory

    for direc in [weight_path, os.getcwd()]:
        stat_path = os.path.join(direc, "test_stats.json")

        with open(stat_path, 'w') as f_open:
            json.dump(final_stats, f_open, sort_keys=True, indent=4)
        log_train(f"Test stats saved in {stat_path}")

    # put the validation loader and the old model back
    trainer.validation_loader = val_loader
    trainer._model = old_model.to(device)


def get_deltas(base_keys):
    deltas = []
    for i, e_i in enumerate(base_keys):
        if i == 0:
            continue
        e_j = base_keys[i-1]
        delta = f"{e_i}_{e_j}_delta"
        deltas.append(delta)

    return deltas


def optim_loss_hooks(model,
                     max_epochs,
                     metric_names,
                     base_keys,
                     grad_keys,
                     weight_path,
                     base,
                     rank,
                     world_size,
                     lr,
                     lr_patience,
                     lr_decay,
                     lr_min,
                     loss_type=None,
                     loss_coef=None,
                     multi_loss_dict=None):
    """
    Make the optimizer, the loss function, and the custom hooks for the trainer.
    Args:
        params (dict): training/network parameters
        model (nff.nn.models): NFF model
        metric_names (list[str]): names of the metrics you want to monitor
        base_keys (list[str]): names of properties the network is predicting
        grad_keys (list[str]): names of any gradients of properites it's
            predicting. Should have the form [key + "_grad"] for all the keys
            you want the gradient of.
        weight_path (str): path to the folder that the model is being trained in
        base (bool): whether this training process has global rank 0
        rank (int): local rank on the node
        world_size (int): total number of processes being used for training
    Returns:
        loss_fn (callable): a function that computes the loss from predictions
            and ground truth
        optimizer (torch.optim.Adam): Adam optimizer instance
        train_hooks (list): list of hooks to apply to the trainer

    """

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=lr)

    loss_fn = build_loss(loss_coef=loss_coef,
                         loss_type=loss_type,
                         multi_loss_dict=multi_loss_dict)

    # make the train metrics
    train_metrics = []
    for metric_name in metric_names:
        metric = getattr(metrics, metric_name)
        deltas = get_deltas(base_keys)
        for key in [*deltas, *base_keys, *grad_keys]:
            train_metrics.append(metric(key))

    # make the train hooks

    train_hooks = [
        hooks.MaxEpochHook(max_epochs),
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer,
            patience=lr_patience,
            factor=lr_decay,
            min_lr=lr_min,
            window_length=1,
            stop_after_min=True
        )
    ]

    # the logging path is the main folder if this is the base,
    # and the rank sub_folder otherwise

    log_path = weight_path if base else (
        os.path.join(weight_path, str(rank)))
    train_hooks.append(
        hooks.PrintingHook(
            log_path,
            metrics=train_metrics,
            separator=' | ',
            world_size=world_size,
            global_rank=rank
        )
    )

    return loss_fn, optimizer, train_hooks


def get_max_batch_iters(weight_path, world_size):
    """
    If using disk-writing parallelization, then the training data
    will be taken from model_path/0/train.pth.tar for process 0,
    model_path/1/train.pth.tar for process 1, etc. These in general
    will have different numbers of species, and so there will be
    different numbers of batches for each process. If they
    are all supposed to wait for the others to finish a batch, but one
    process can't finish it because it's already out of data, the training
    will hang. This function finds the maximum number of batches that any
    process can finish without doing more than any of the other processes.

    Args:
        weight_path (str): path to the folder that the model is being trained in
        world_size (int): total number of processes being used for training
    Returns:
        max_batch_iters (int): maximum number of batches for any parallel process
    """

    max_batch_iters = float("inf")

    # go through each folder and load "train_len" once it's available, which
    # is the number of batches in the training set. Take the smallest value

    for rank in range(world_size):
        batch_path = os.path.join(weight_path, str(rank), "train_len")
        while True:
            try:
                with open(batch_path, "r") as f:
                    batch_iters = int(f.read().strip())
                break
            except (FileNotFoundError, ValueError):
                continue
        if batch_iters < max_batch_iters:
            max_batch_iters = batch_iters
    return max_batch_iters


def get_trainer_kwargs(params, weight_path, world_size):
    """
    Get any extra arguments for the trainer class that may have been requested.
    Args:
        params (dict): training/network parameters
        weight_path (str): path to the folder that the model is being trained in
        world_size (int): total number of processes being used for training
    Returns:
        trainer_kwargs (dict): dictionary of keyword arguments
    """

    max_batch_iters = get_max_batch_iters(weight_path=weight_path,
                                          world_size=world_size)

    trainer_kwargs = dict(
        max_batch_iters=max_batch_iters,
        # any kwargs when calling the model
        model_kwargs=params.get("model_kwargs"),
        # how many batches to accumulate gradients over before
        # taking an optimization step
        mini_batches=params.get("mini_batches", 1),
        # how many checkpoints to keep
        checkpoints_to_keep=params.get(
            "checkpoints_to_keep", 3),
        # how often to make checkpoints
        checkpoint_interval=params.get(
            "checkpoint_interval", 1),
        # normalize loss by molecule not by atom number
        mol_loss_norm=params.get("mol_loss_norm",
                                 False),
        # how often to delete pickle files with gradients
        del_grad_interval=params.get("del_grad_interval",
                                     10),
        # use a metric instead of the loss to determine the
        # trainer scheduling
        metric_as_loss=params.get("metric_as_loss"),
        # whether you want to maximize or minimize that metric
        metric_objective=params.get("metric_objective"),
        # cut off an epoch after `epoch_cutoff` batches. - useful if you want
        # to validate the model more often than after going through all data
        # points once.
        epoch_cutoff=params.get("epoch_cutoff", float("inf"))
    )

    return trainer_kwargs


def load_params(file):
    """
    Load the train config parameters.
    Args:
        file (str): path to config file
    Returns:
        out (dict): config dictionary
    """
    with open(file, "r") as f_open:
        out = json.load(f_open)
    return out


def build_loss(loss_coef,
               loss_type,
               multi_loss_dict):

    if multi_loss_dict is not None:
        loss_fn = loss.build_multi_loss(multi_loss_dict)
    elif loss_coef is not None:
        loss_builder = getattr(loss, "build_{}_loss".format(loss_type))
        loss_fn = loss_builder(loss_coef=loss_coef)
    else:
        raise Exception("Must specify either `loss_coef` or `multi_loss_dict`")

    return loss_fn


def plural(key):
    if key.endswith("s"):
        plural_key = key + "es"
    else:
        plural_key = key + "s"
    return plural_key


def get_train_params(params):

    main_keys = ['loss',
                 'loss_coef',
                 'multi_loss_dict']

    other_keys = ['lr',
                  'lr_patience',
                  'lr_decay',
                  'lr_min',
                  'max_epochs']

    train_params = {}

    for key in main_keys:
        if key not in params:
            continue
        val = params[key]
        if key == 'loss_coef' and isinstance(key, str):
            val = json.loads(val)

        plural_key = plural(key)
        if isinstance(val, list):
            train_params[plural_key] = val
        else:
            train_params[plural_key] = [val]

    use_keys = [plural(key) for key in main_keys if plural(key) in
                train_params]
    num_types = max([len(train_params[use_key]) for use_key in use_keys])
    # get rid of any keys that don't have the maximum length (e.g. if for
    # some reason you have `loss_coef` but also have `multi_loss_dict` of
    # length 2)

    for use_key in use_keys:
        if use_key not in train_params:
            continue
        if len(train_params[use_key]) < num_types:
            train_params.pop(use_key)

    for key in other_keys:
        val = params[key]
        if isinstance(val, list):
            train_params[key] = val
        else:
            train_params[key] = [val] * num_types

    return train_params, num_types


def train_sequential(weight_path,
                     model,
                     loader_sets,
                     # train_loader,
                     # val_loader,
                     world_size,
                     rank,
                     trainer_kwargs,
                     params,
                     metric_names,
                     base_keys,
                     grad_keys,
                     base,
                     device,
                     reset_trainer,
                     log_train,
                     sequential_best):

    # we need the option for different samplers in different
    # stages of the training

    train_params, num_types = get_train_params(params)
    loss_coefs = train_params.get("loss_coefs", [None] * num_types)
    loss_types = train_params.get("losses", [None] * num_types)
    multi_loss_dicts = train_params.get("multi_loss_dicts",
                                        [None] * num_types)
    if len(loader_sets) == 1:
        loader_sets *= num_types
    elif len(loader_sets) != num_types:
        raise Exception("Specified %d sampler types for "
                        "%d different sequences"
                        % (len(loader_sets),  num_types))

    param_path = os.path.join(weight_path, "params.json")
    with open(param_path, 'w') as f_open:
        json.dump(params, f_open, sort_keys=True, indent=4)
    log_train(f"Model and training details saved in {param_path}")

    for i in range(num_types):
        # once for putting into the trainer and once for
        # keeping unused so it won't get written over during
        # load_state_dict in the trainer
        loss_fns = []
        optimizers = []
        hooks_list = []

        for _ in range(2):
            loss_fn, optimizer, train_hooks, = optim_loss_hooks(
                model,
                max_epochs=train_params["max_epochs"][i],
                metric_names=metric_names,
                base_keys=base_keys,
                grad_keys=grad_keys,
                weight_path=weight_path,
                base=base,
                rank=rank,
                world_size=world_size,
                lr=train_params["lr"][i],
                lr_patience=train_params["lr_patience"][i],
                lr_decay=train_params["lr_decay"][i],
                lr_min=train_params["lr_min"][i],
                loss_type=loss_types[i],
                loss_coef=loss_coefs[i],
                multi_loss_dict=multi_loss_dicts[i])

            loss_fns.append(loss_fn)
            optimizers.append(optimizer)
            hooks_list.append(train_hooks)

        train_loader = loader_sets[i][0]
        val_loader = loader_sets[i][1]

        trainer = Trainer(
            model_path=weight_path,
            model=model,
            loss_fn=loss_fns[0],
            optimizer=optimizers[0],
            train_loader=train_loader,
            validation_loader=val_loader,
            hooks=hooks_list[0],
            world_size=world_size,
            global_rank=rank,
            **trainer_kwargs)

        if i != 0 or reset_trainer:
            trainer.optimizer = optimizers[1]
            trainer.loss_fn = loss_fns[1]
            trainer.best_loss = float("inf")
            trainer.hooks = hooks_list[1]

        trainer._model.add_nacv = params.get("add_nacv", False)
        seq_dir = os.path.join(weight_path, f"sequence_{i}")
        do_train = not os.path.isdir(seq_dir)
        if not do_train:
            continue

        if i != 0 and sequential_best:
            model_type = params['model_type']
            old_seq_dir = os.path.join(weight_path, f"sequence_{i-1}")
            best_model = load_model(old_seq_dir,
                                    params=params,
                                    model_type=model_type)
            state_dict = best_model.state_dict()
            trainer._model.load_state_dict(state_dict)

        trainer.train(device=device,
                      n_epochs=train_params["max_epochs"][i])

        os.makedirs(seq_dir)
        files = ['log_human.read.csv', 'best_model', 'best_model.pth.tar']
        for file in files:
            file_path = os.path.join(weight_path, file)
            if not os.path.isfile(file_path):
                continue
            new_path = os.path.join(seq_dir, file)
            if os.path.isfile(new_path):
                os.remove(new_path)
            shutil.copy(file_path, new_path)

        folders = ['checkpoints']
        for folder in folders:
            folder_path = os.path.join(weight_path, folder)
            if not os.path.isdir(folder_path):
                continue
            new_path = os.path.join(seq_dir, folder)
            if os.path.isdir(new_path):
                shutil.rmtree(new_path)
            shutil.copytree(folder_path, new_path)

    return trainer


def train(gpu,
          all_params,
          world_size,
          node_rank,
          gpus,
          metric_names,
          base_keys,
          grad_keys,
          needs_nbrs,
          needs_angles,
          cutoff,
          reset_trainer,
          sequential_best):
    """
    Train a model in parallel.
    Args:
        gpu (int): index of the current gpu
        all_params (dict): job_info dictionary
        world_size (int): total number of gpus
        node_rank (int): index of the current node
        gpus (int): number of gpus per node
        metric_names (list[str]): metrics to monitor
        base_keys (list[str]): keys that the model is
            directly predicting
        grad_keys (list[str]): gradients of quantities
            that the model is spredicting
    Returns:
        None
    """

    # get the neural network quantities
    model_name, params, weight_path = get_nn_quants(all_params)

    # get the parallel quantities
    rank, batch_size, base, log_train = init_quants(node_rank=node_rank,
                                                    gpu=gpu,
                                                    gpus=gpus,
                                                    world_size=world_size,
                                                    params=params)

    log_train('neural network id: ' + str(model_name))
    log_train("Making loaders...")

    loader_sets = make_all_loaders(weight_path=weight_path,
                                   rank=rank,
                                   world_size=world_size,
                                   params=params,
                                   base=base,
                                   batch_size=batch_size,
                                   node_rank=node_rank,
                                   gpu=gpu,
                                   gpus=gpus,
                                   log_train=log_train,
                                   needs_nbrs=needs_nbrs,
                                   needs_angles=needs_angles,
                                   cutoff=cutoff)

    # train_loader, val_loader, test_loader = loaders

    log_train("Created loaders.")
    log_train("Setting up training...")

    # if the world size is 1, then allow the option of `device` being
    # something other than `gpu` (e.g. if you want to train on a cpu)

    if world_size == 1:
        device = params.get("device", 0)
    else:
        device = gpu

    model = make_model(params=params,
                       device=device,
                       world_size=world_size,
                       weight_path=weight_path,
                       log_train=log_train)

    trainer_kwargs = get_trainer_kwargs(params=params,
                                        weight_path=weight_path,
                                        world_size=world_size)

    trainer = train_sequential(weight_path=weight_path,
                               model=model,
                               loader_sets=loader_sets,
                               # train_loader=train_loader,
                               # val_loader=val_loader,
                               world_size=world_size,
                               rank=rank,
                               trainer_kwargs=trainer_kwargs,
                               params=params,
                               metric_names=metric_names,
                               base_keys=base_keys,
                               grad_keys=grad_keys,
                               base=base,
                               device=device,
                               reset_trainer=reset_trainer,
                               log_train=log_train,
                               sequential_best=sequential_best)

    log_train('model saved in ' + weight_path)

    # make test stats

    val_loader = loader_sets[-1][1]
    test_loader = loader_sets[-1][2]

    make_stats(trainer=trainer,
               test_loader=test_loader,
               params=params,
               weight_path=weight_path,
               device=device,
               val_loader=val_loader,
               log_train=log_train)


def add_args(all_params):
    """
    Add arguments when calling `train` that can be found in `all_params`.
    Args:
        all_params (dict): config dictionary
    Returns:
        args (list): extra arguments in `train`
    """

    params = {**all_params,
              **all_params.get('train_params', {}),
              **all_params.get('model_params', {}),
              **all_params.get('details', {})}
    metric_names = params.get("metrics", DEFAULT_METRICS)
    base_keys = params.get("base_keys", params.get("output_keys", ["energy"]))
    grad_keys = params.get("grad_keys", ["energy_grad"])
    needs_nbrs = params.get("needs_nbrs", True)
    needs_angles = params.get("needs_angles", False)
    cutoff = params.get("cutoff", DEFAULT_CUTOFF)
    reset_trainer = params.get("reset_trainer", False)
    sequential_best = params.get("sequential_best", True)

    args = [metric_names,
            base_keys,
            grad_keys,
            needs_nbrs,
            needs_angles,
            cutoff,
            reset_trainer,
            sequential_best]

    return args


def main():

    parser = argparse.ArgumentParser(description="Trains a neural potential")
    parser.add_argument('paramsfile', type=str, default=DEFAULTPARAMSFILE,
                        help="file containing all parameters")
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--node_rank', default=0, type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()
    world_size = args.gpus * args.nodes
    params = load_params(args.paramsfile)
    extra_args = add_args(params)

    if world_size == 1:
        train(0,
              params,
              world_size,
              0,
              1,
              *extra_args)
    else:
        mp.spawn(train, nprocs=args.gpus,
                 args=(params,
                       world_size,
                       args.node_rank,
                       args.gpus,
                       *extra_args))


if __name__ == "__main__":
    main()
