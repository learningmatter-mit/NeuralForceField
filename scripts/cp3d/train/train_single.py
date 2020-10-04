import os
import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from nff.data import Dataset, split_train_validation_test, collate_dicts
from nff.data.loader import ImbalancedDatasetSampler
from nff.train import metrics, Trainer, get_model, loss, hooks

import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

DEFAULTPARAMSFILE = 'job_info.json'
DEFAULT_METRIC = "MeanAbsoluteError"


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


def get_gpu_splits(weight_path, rank, world_size, params):
    """ 
    Check if there are already datasets in each parallel folder.
    If so, load and return those datasets instead of loading the whole
    thing in memory on every gpu and splitting afterwards.

    Args:
        weight_path (str): training folder
        rank (int): global rank of the current process
        world_size (int): total number number of gpus altogether
        params (dict): training/network parameters

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
        datasets = [Dataset.from_file(path) for path in split_paths]
        return datasets

    # otherwise get the dataset, split it, and save it

    dataset = Dataset.from_file(dat_path)

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


def get_sampler(params,
                dataset,
                rank,
                world_size,
                central_data,
                custom_sampler,
                torch_par):
    """
    Get sampler for instantiating the DataLoader.
    Args:
        params (dict): training/network parameters
        dataset (nff.data.Dataset): dataset to load
        rank (int): global rank of the current process
        world_size (int): total number number of gpus altogether
        central_data (bool): whether the data is centralized in one
            location or divided into GPU folders.
        custom_sampler (bool): whether to use a custom sampler
            (True for train, False for val and test)
        torch_par (bool): whether or not to use torch parallelization
            (alternative is just writing gradients to disk and loading
            them)
    Returns:
        sample: the Data sampler

    Example:
        1. Asking for a sampler for imbalanced data 
            params = {"sampler": {"name": "ImbalancedDataSampler",
                                  "target_name": "bind"}}

    """

    # See if any specific sampler is mentioned in `params`

    sample_params = params.get("sampler", {})

    if not central_data:
        if not custom_sampler:
            sampler = None
        elif sample_params.get("name") == "ImbalancedDatasetSampler":
            target_name = sample_params["target_name"]
            sampler = ImbalancedDatasetSampler(target_name=target_name,
                                               props=dataset.props)
        elif sample_params.get("name") is not None:
            raise NotImplementedError(("Sampler {} not yet "
                                       "implemented".format(
                                           sample_params["name"])))
        else:
            sampler = None

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

    return sampler


def make_loader(dataset,
                params,
                batch_size,
                world_size,
                rank,
                weight_path,
                central_data,
                custom_sampler,
                torch_par):
    """
    Make distributed loader for a dataset.
    Args:
        dataset (nff.data.Dataset): dataset to load
        batch_size (int): per-gpu batch size (so the
            effective batch size is batch_size x world_size)
        world_size (int): total number number of gpus altogether
        rank (int): global rank of the current process
        weight_path (str): training folder
        central_data (bool): whether the data is centralized in one
            location or divided into GPU folders.
        custom_sampler (bool): whether to use a custom sampler
            (True for train, False for val and test)
        torch_par (bool): whether or not to use torch parallelization
            (alternative is just writing gradients to disk and loading
            them)

    Returns:
        loader (torch.utils.data.DatalLoader): data loader
            for the dataset.

    """

    # get the dictionary with information about the sampler to be used
    # in the loader

    sampler = get_sampler(params=params,
                          dataset=dataset,
                          rank=rank,
                          world_size=world_size,
                          central_data=central_data,
                          custom_sampler=custom_sampler,
                          torch_par=torch_par)

    # make the loader with the custom collate function,
    # and shuffle=False, and any sampler information

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_dicts,
        shuffle=False,
        sampler=sampler)

    return loader


def make_all_loaders(weight_path,
                     rank,
                     world_size,
                     params,
                     base,
                     model_name,
                     batch_size,
                     node_rank,
                     gpu,
                     gpus,
                     log_train):
    """
    Get train, val, and test data loaders.
    Args:
        weight_path (str): training folder
        rank (int): global rank of the current process
        world_size (int): total number number of gpus altogether
        params (dict): training/network parameters
        base (bool): whether this is the base process
        geoms (QuerySet): Geom objects to use when creating the dataset
        model_name (int): name given to the model
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

    gpu_splits = get_gpu_splits(weight_path=weight_path,
                                rank=rank,
                                world_size=world_size,
                                params=params)

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
            train, val, test = make_or_get_datasets(params=params,
                                                    model_name=model_name,
                                                    weight_path=weight_path)
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
        train, val, test = get_datasets(weight_path)

    # record dataset stats

    d_set_size = len(train) + len(val) + len(test)
    log_train('{} data points'.format(d_set_size))

    # create data loaders

    loaders = []
    for i, d_set in enumerate([train, val, test]):

        custom_sampler = True if (i == 0) else False
        loader = make_loader(dataset=d_set,
                             params=params,
                             batch_size=batch_size,
                             world_size=world_size,
                             rank=rank,
                             weight_path=weight_path,
                             central_data=central_data,
                             custom_sampler=custom_sampler,
                             torch_par=torch_par)

        loaders.append(loader)

    train_len = len(loaders[0])
    len_path = os.path.join(weight_path, str(rank), "train_len")
    with open(len_path, "w") as f:
        f.write(str(train_len))

    return loaders


def dsets_from_folder(weight_path):
    """
    Load train, val, and test datasets from the main folder.
    Args:
        weight_path (str): training folder
    Returns:
        datasets (list): train, val, and test datasets

    """

    names = ['train', 'val', 'test']
    datasets = []
    for name in names:
        data_path = os.path.join(weight_path, "{}.pth.tar".format(name))
        if not os.path.isfile(data_path):
            break
        datasets.append(Dataset.from_file(data_path))

    if len(datasets) == 3:
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


def make_or_get_datasets(model_name, params, weight_path):
    """
    Create the nff dataset and split it into train, val, and test.
    If the dataset is already there (e.g. if you've picked up
    training where you left off earlier), then load it.
    Args:
        params (dict):   training/network parameters
        model_name (int): neural network database object ID
        weight_path (str): training folder
    Returns:
        datasets (list): train, val and test datasets
    """

    datasets = dsets_from_folder(weight_path)
    return datasets


def get_datasets(weight_path):
    """
    Load the datasets.
    Args:
        weight_path (str): training folder
    Returns:
        datasets (list): train, val and test datasets

    """

    datasets = []
    names = ['train', 'val', 'test']

    for name in names:
        data_path = os.path.join(weight_path, "{}.pth.tar".format(name))
        datasets.append(Dataset.from_file(data_path))

    return datasets


def make_model(params, device, world_size, weight_path):
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

    start_model_path = os.path.join(weight_path, "start_model")

    if os.path.isfile(start_model_path):
        model = torch.load(start_model_path, map_location="cpu")
        print(f"Loading model from {start_model_path}")
        sys.stdout.flush()
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
        model_name (int): id of the neural network
        params (dict): job params
        nn (neuralnet.models.NnPotential): neural network database object
        geoms (QuerySet): Geom objects to use when creating the dataset
        weight_path (str): training folder

    """

    params = {**all_params['train_params'],
              **all_params['model_params']}
    model_name = params['model_name']
    weight_path = os.path.join(params['weightpath'], str(model_name))

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


def get_best_model(T):
    """
    Get the best model from a trainer.
    Args:
        T (Trainer): trainer instance
    Returns:
        best_model (nff.nn.model): best model
    """

    # continue looping until you succesfully
    # get the best model
    while True:
        try:
            best_model = T.get_best_model()
            return best_model
        except (EOFError, FileNotFoundError):
            continue


def make_stats(T,
               test_loader,
               loss_fn,
               params,
               weight_path,
               global_rank,
               base,
               world_size,
               device,
               base_keys,
               grad_keys,
               trainer_kwargs,
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

    # set the trainer validation loader to the test
    # loader so you can use its metrics to get the
    # performance on the test set

    T.validation_loader = test_loader

    # validate on test loader
    T.validate(device, test=True)

    # get the metric performance

    log_hook = [h for h in T.hooks if isinstance(h, hooks.PrintingHook)][0]
    final_stats = log_hook.aggregate(T, test=True)

    stat_path = os.path.join(weight_path, "test_stats.json")
    param_path = os.path.join(weight_path, "params.json")

    with open(stat_path, 'w') as f:
        json.dump(final_stats, f, sort_keys=True, indent=4)

    with open(param_path, 'w') as f:
        json.dump(params, f, sort_keys=True, indent=4)

    log_train(f"Test stats saved in {stat_path}")
    log_train(f"Model and training details saved in {param_path}")

    T.validation_loader = val_loader


def optim_loss_hooks(params,
                     model,
                     metric_names,
                     base_keys,
                     grad_keys,
                     weight_path,
                     base,
                     rank,
                     world_size):

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=params['lr'])

    # loss function and training metrics
    loss_coef = params['loss_coef']
    if isinstance(loss_coef, str):
        loss_coef = json.loads(loss_coef)

    loss_type = params.get("loss", "mse")
    loss_builder = getattr(loss, "build_{}_loss".format(loss_type))
    loss_fn = loss_builder(loss_coef=loss_coef)

    train_metrics = []
    for metric_name in metric_names:
        metric = getattr(metrics, metric_name)
        for key in [*base_keys, *grad_keys]:
            train_metrics.append(metric(key))

    train_hooks = [
        hooks.MaxEpochHook(params['max_epochs']),
        hooks.ReduceLROnPlateauHook(
            optimizer=optimizer,
            patience=params.get('lr_patience'),
            factor=params.get('lr_decay'),
            min_lr=params.get('lr_min'),
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

    max_batch_iters = float("inf")
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

    max_batch_iters = get_max_batch_iters(weight_path=weight_path,
                                          world_size=world_size)

    trainer_kwargs = dict(
        max_batch_iters=max_batch_iters,
        model_kwargs=params.get("model_kwargs"),
        mini_batches=params.get("mini_batches", 1),
        checkpoints_to_keep=params.get(
            "checkpoints_to_keep", 3),
        checkpoint_interval=params.get(
            "checkpoint_interval", 1),
        mol_loss_norm=params.get("mol_loss_norm",
                                 False),
        del_grad_interval=params.get("del_grad_interval",
                                     10),
        metric_as_loss=params.get("metric_as_loss"),
        metric_objective=params.get("metric_objective"),
        epoch_cutoff=params.get("epoch_cutoff", float("inf"))
    )

    return trainer_kwargs


def load_params(file):
    with open(file, "r") as f:
        out = json.load(f)
    return out


def train(gpu,
          all_params,
          world_size,
          node_rank,
          gpus,
          metric_names,
          base_keys,
          grad_keys):
    """
    Train a model in parallel.
    Args:
        gpu (int): index of the current gpu
        all_params (dict): job_info dictionary
        world_size (int): total number of gpus
        node_rank (int): index of the current node
        gpus (int): number of gpus per node
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

    loaders = make_all_loaders(weight_path=weight_path,
                               rank=rank,
                               world_size=world_size,
                               params=params,
                               base=base,
                               model_name=model_name,
                               batch_size=batch_size,
                               node_rank=node_rank,
                               gpu=gpu,
                               gpus=gpus,
                               log_train=log_train)
    train_loader, val_loader, test_loader = loaders

    log_train("Created loaders.")
    log_train("Setting up training...")

    # if the world size is 1, then allow the option of `device` being
    # something other than `gpu` (e.g. if you want to train on a cpu)

    if world_size == 1 and params.get("device") == "cpu":
        device = "cpu"
    else:
        device = gpu

    model = make_model(params=params,
                       device=device,
                       world_size=world_size,
                       weight_path=weight_path)

    loss_fn, optimizer, train_hooks = optim_loss_hooks(
        params=params,
        model=model,
        metric_names=metric_names,
        base_keys=base_keys,
        grad_keys=grad_keys,
        weight_path=weight_path,
        base=base,
        rank=rank,
        world_size=world_size
    )

    trainer_kwargs = get_trainer_kwargs(params=params,
                                        weight_path=weight_path,
                                        world_size=world_size)

    log_train("Training...")

    T = Trainer(
        model_path=weight_path,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
        hooks=train_hooks,
        world_size=world_size,
        global_rank=rank,
        **trainer_kwargs
    )

    save_path = os.path.join(weight_path, str(node_rank * gpus + gpu),
                             "init_model.pth.tar")
    torch.save(model, save_path)
    T.train(device=device, n_epochs=params['max_epochs'])

    log_train('model saved in ' + weight_path)

    make_stats(T=T,
               test_loader=test_loader,
               loss_fn=loss_fn,
               params=params,
               weight_path=weight_path,
               global_rank=rank,
               base=base,
               world_size=world_size,
               device=device,
               base_keys=base_keys,
               grad_keys=grad_keys,
               trainer_kwargs=trainer_kwargs,
               val_loader=val_loader,
               log_train=log_train)


def add_args(all_params):

    params = {**all_params['train_params'],
              **all_params['model_params']}
    metric_names = params.get("metrics", [DEFAULT_METRIC])
    base_keys = params.get("base_keys", ["energy"])
    grad_keys = params.get("grad_keys", ["energy_grad"])

    args = [metric_names,
            base_keys,
            grad_keys]

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
