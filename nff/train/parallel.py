"""
Tools to implement parallelization by writing quantities to disk and loading them
between processes.
"""

import pickle
import os


def get_grad(optimizer):

    grad_list = []
    for group in optimizer.param_groups:
        grad_list.append([])
        for param in group['params']:
            if param.grad is None:
                grad_list[-1].append(param.grad)
            else:
                grad_list[-1].append(param.grad.detach().cpu())
    return grad_list


def save_grad(optimizer,
              loss_size,
              rank,
              weight_path,
              batch_num,
              epoch):

    grad_list = get_grad(optimizer)
    save_dic = {"grad": grad_list, "loss_size": loss_size}
    save_path = os.path.join(weight_path, str(rank),
                             "grad_{}_{}.pickle".format(
        epoch, batch_num))
    with open(save_path, "wb") as f:
        pickle.dump(save_dic, f)


def add_grads(optimizer,
              loss_size,
              weight_path,
              rank,
              world_size,
              batch_num,
              epoch,
              device):

    # Set the optimizer to have zero gradient and then load in all
    # the grads. This ensures no differences in the gradients between the
    # different processes, which would occur due to loss of precision
    # if the numbers were added in different orders.

    optimizer.zero_grad()

    # paths to all pickle files

    paths = [os.path.join(weight_path, str(index),
                          "grad_{}_{}.pickle".format(epoch, batch_num))
             for index in range(world_size)]

    loaded_grads = {path: None for path in paths}

    while None in loaded_grads.values():
        missing_paths = [key for key, val
                         in loaded_grads.items() if val is None]
        for path in missing_paths:
            try:
                with open(path, "rb") as f:
                    loaded_grads[path] = pickle.load(f)
            except (EOFError, FileNotFoundError, pickle.UnpicklingError):
                continue

    # total size is the sum of all sizes from each process
    total_size = sum([grad_dic["loss_size"] for
                      grad_dic in loaded_grads.values()]
                     )

    for k, grad_dic in enumerate(loaded_grads.values()):
        for i, group in enumerate(optimizer.param_groups):
            for j, param in enumerate(group['params']):
                if param.grad is None:
                    continue
                param.grad += grad_dic["grad"][i][j].to(device)

                # if you're at the last grad_dic, divide
                # by the total size

                if k == len(loaded_grads.values()) - 1:
                    param.grad /= total_size

    return optimizer


def del_grad(rank,
             epoch,
             batch_num,
             weight_path,
             del_interval,
             max_batch_iters):

    # epoch starts counting from 1 and batch_num starts
    # counting from 0

    num = (epoch - 1) * max_batch_iters + batch_num + 1
    if num % del_interval == 0:
        folder = os.path.join(weight_path, str(rank))
        for file in os.listdir(folder):
            if file.startswith("grad") and file.endswith("pickle"):
                this_epoch = int(file.split("_")[1])
                # remove things not from this epoch
                if this_epoch != epoch:
                    file_path = os.path.join(folder, file)
                    os.remove(file_path)


def update_optim(optimizer,
                 loss_size,
                 rank,
                 world_size,
                 weight_path,
                 batch_num,
                 epoch,
                 del_interval,
                 device,
                 max_batch_iters):

    save_grad(optimizer=optimizer,
              loss_size=loss_size,
              rank=rank,
              weight_path=weight_path,
              batch_num=batch_num,
              epoch=epoch)

    optimizer = add_grads(optimizer=optimizer,
                          loss_size=loss_size,
                          weight_path=weight_path,
                          rank=rank,
                          world_size=world_size,
                          batch_num=batch_num,
                          epoch=epoch,
                          device=device)
    del_grad(rank=rank,
             epoch=epoch,
             batch_num=batch_num,
             weight_path=weight_path,
             del_interval=del_interval,
             max_batch_iters=max_batch_iters)

    return optimizer
