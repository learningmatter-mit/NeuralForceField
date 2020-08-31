from torch.utils.data import Dataset as TorchDataset
from torch.nn import functional as F
import torch
import random
import numpy as np
import os
import argparse
import json
import sys
import pdb
import copy

from nff.data import Dataset as NffDataset

SIM_FUNCS = {"cosine_similarity": F.cosine_similarity}
PROP = "sars_cov_one_cl_protease_active"
SMILES_PATH = "/home/saxelrod/rgb_nfs/GEOM_DATA_ROUND_2/mc_smiles.json"


class ConfDataset(TorchDataset):

    def __init__(self, nff_dset, seed=None):
        self.props = self.make_props(nff_dset, seed)

    def get_conf_idx(self, seed, num_confs):

        if seed is not None:
            torch.manual_seed(seed)

        conf_idx = [torch.randint(num_conf, (1,)).item()
                    for num_conf in num_confs]
        torch.seed()

        return conf_idx

    def randomize_conf_fps(self, seed):

        conf_idx = self.get_conf_idx(seed, self.props["num_confs"])
        all_fps = self.props["all_fps"]
        single_fps = [fp[idx] for fp, idx in zip(all_fps, conf_idx)]

        self.props["conf_idx"] = conf_idx
        self.props["single_fps"] = single_fps

    def get_att_weights(self, conf_idx, num_confs):

        all_weights = []
        for idx, num in zip(conf_idx, num_confs):
            weights = torch.zeros(num)
            weights[idx] = 1
            all_weights.append(weights)
        return all_weights

    def make_props(self, nff_dset, seed):

        num_confs = [len(batch["weights"]) for batch
                     in nff_dset]
        conf_idx = self.get_conf_idx(seed, num_confs)
        all_fps = nff_dset.props["fingerprint"]
        single_fps = [fp[idx] for fp, idx in zip(all_fps, conf_idx)]
        att_weights = self.get_att_weights(conf_idx, num_confs)

        props = {"num_confs": num_confs,
                 "conf_idx": conf_idx,
                 "att_weights": att_weights,
                 "all_fps": all_fps,
                 "single_fps": single_fps}

        return props

    def flip_fp(self, spec_idx, conf_idx, reset_att=True):

        new_fp = self.props["all_fps"][spec_idx][conf_idx]
        self.props["single_fps"][spec_idx] = new_fp
        self.props["conf_idx"][spec_idx] = conf_idx

        self.props["att_weights"][spec_idx] *= 0
        self.props["att_weights"][spec_idx][conf_idx] = 1

    def compare(self, spec_idx, func_name, other_dset=None):

        if other_dset is not None:
            other_fps = torch.stack(other_dset.props["single_fps"])
            length = len(other_dset)
        else:
            other_fps = torch.stack(self.props["single_fps"])
            length = len(self)

        repeat_fp = self.props["single_fps"][spec_idx].repeat(length, 1)

        sim_func = SIM_FUNCS[func_name]
        sim = sim_func(repeat_fp, other_fps)

        return sim

    def __len__(self):
        return len(self.props['all_fps'])

    def __getitem__(self, idx):

        return {key: val[idx] for key, val in self.props.items()}

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def from_file(cls, path):
        obj = torch.load(path)
        if isinstance(obj, cls):
            return obj
        else:
            raise TypeError(
                '{} is not an instance from {}'.format(path, type(cls))
            )


def choose_flip(dataset):
    """
    Args:
        num_confs_list (list): INFO
    Example:
        num_confs_list = [4, 10, 30, 100]
    """

    conf_list = dataset.props["num_confs"]

    # choose the species to change
    num_specs = len(conf_list)
    spec_idx = random.randint(0, num_specs - 1)

    # choose the new conformer to assign
    num_confs = conf_list[spec_idx]
    conf_idx = random.randint(0, num_confs - 1)

    return spec_idx, conf_idx


def get_loss_change(loss_mat,
                    spec_idx,
                    func_name,
                    dataset,
                    other_dset,
                    max_sim):

    # the new similarity between the species of interest
    # and all other species' fingeprints

    new_sim = dataset.compare(spec_idx, func_name, other_dset)
    denom = len(dataset) * len(other_dset)

    # import pdb
    # pdb.set_trace()

    if max_sim:
        add_loss = - new_sim.sum() / denom
    else:
        add_loss = new_sim.sum() / denom

    subtract_loss = loss_mat[spec_idx, :].sum() / denom

    delta_loss = add_loss - subtract_loss

    return delta_loss, new_sim


def init_loss(dataset,
              func_name,
              other_dset,
              max_sim):
    """


    Args:
        max_sim (bool): if True, minimizing the loss maximizes 
            the similarity. If False, minimizing the loss minimizes
            the similarity.
    """

    len_0 = len(dataset)
    len_1 = len(other_dset)
    loss_mat = torch.zeros((len_0, len_1))

    for spec_idx in range(len_0):

        sim = dataset.compare(spec_idx, func_name, other_dset)
        if max_sim:
            # loss_mat[:, spec_idx] = -sim
            loss_mat[spec_idx, :] = -sim
        else:
            # loss_mat[:, spec_idx] = sim
            loss_mat[spec_idx, :] = sim

    import pdb
    # pdb.set_trace()

    loss = loss_mat.mean().item()

    return loss_mat, loss


def update_loss(loss,
                loss_mat,
                spec_idx,
                delta_loss,
                new_sim,
                max_sim):

    if max_sim:
        # loss_mat[:, spec_idx] = -new_sim
        loss_mat[spec_idx, :] = -new_sim
    else:
        # loss_mat[:, spec_idx] = new_sim
        loss_mat[spec_idx, :] = new_sim

    loss += delta_loss

    return loss_mat, loss


def get_criterion(delta_loss, temp):

    if delta_loss < 0:
        return True

    p = torch.exp(-delta_loss / temp)
    r = random.random()

    return p > r


def make_exp_sched(start_temp, num_temps, end_temp, dset_len):

    def temp_func():

        tau = (num_temps - 1) / np.log(start_temp / end_temp)
        intervals = np.arange(num_temps)
        temps = start_temp * np.exp(-intervals / tau)

        return temps

    return temp_func


def make_temp_func(temp_dic, dset_len):

    name = temp_dic["name"]
    params = temp_dic["params"]

    if name == "exponential":

        start_temp = params["start_temp"]
        end_temp = params["end_temp"]
        num_temps = params["num_temps"]

        temp_func = make_exp_sched(start_temp,
                                   num_temps,
                                   end_temp,
                                   dset_len)
        return temp_func

    else:
        raise NotImplementedError


def fprint(msg, verbose=True):
    if verbose:
        print(msg)
        sys.stdout.flush()


def run_mc(dataset,
           func_name,
           num_sweeps,
           temp_func,
           max_sim,
           other_dset=None,
           update_other=None,
           verbose=False):

    loss_mat, loss = init_loss(dataset, func_name, other_dset, max_sim)
    temps = temp_func()
    num_temps = len(temps)

    fprint(f"Starting loss is %.6e." % loss, verbose)

    # import pdb
    # pdb.set_trace()

    for i, temp in enumerate(temps):

        fprint((f"Starting iteration {i+1} of {num_temps} at "
                "temperature %.3e... " % temp), verbose)

        num_iters = int(len(dataset) * num_sweeps)
        for it in range(num_iters):

            #######
            # actual_loss_mat, actual_loss = init_loss(dataset, func_name)
            #######

            spec_idx, conf_idx = choose_flip(dataset)
            old_conf_idx = dataset.props["conf_idx"][spec_idx]
            dataset.flip_fp(spec_idx, conf_idx)

            # old_dset = copy.deepcopy(dataset)

            if update_other is not None:
                other_dset = update_other(other_dset, dataset)

            # dic = {}
            # for key in old_dset.props.keys():
            #     val_0 = old_dset.props[key]
            #     val_1 = dataset.props[key]
            #     if isinstance(val_0[0], torch.Tensor):
            #         results = all([(i == j).all() for i, j in zip(val_0, val_1)])
            #     else:
            #         results = all([(i == j) for i, j in zip(val_0, val_1)])
            #     dic[key] = results

            # print(dic)
            # import pdb
            # pdb.set_trace()

            # this doesn't work anymore because all the conformers of other_dset
            # have changed. Hence we just need to calculate the loss anew.

            if update_other is None:
                delta_loss, new_sim = get_loss_change(loss_mat,
                                                      spec_idx,
                                                      func_name,
                                                      dataset,
                                                      other_dset,
                                                      max_sim)
                criterion = get_criterion(delta_loss, temp)

                if criterion:
                    loss_mat, loss = update_loss(loss,
                                                 loss_mat,
                                                 spec_idx,
                                                 delta_loss,
                                                 new_sim,
                                                 max_sim)

                else:
                    dataset.flip_fp(spec_idx, old_conf_idx)

            else:
                # this should be working now...?

                new_loss_mat, new_loss = init_loss(
                    dataset, func_name, other_dset, max_sim)

                delta_loss = torch.Tensor([new_loss - loss]).squeeze()
                criterion = get_criterion(delta_loss, temp)

                if criterion:
                    loss_mat = new_loss_mat
                    loss = new_loss
                else:
                    dataset.flip_fp(spec_idx, old_conf_idx)

                if not max_sim:
                    new_actual_loss_mat, new_actual_loss = init_loss(
                        dataset, func_name, other_dset, max_sim)
                    # real_delta = new_actual_loss - actual_loss

                    fprint("Supposed loss: %.6e" % loss)
                    fprint("Real loss: %.6e" % new_actual_loss)


            # fprint(f"Completed iteration {it+1} of {num_iters}", verbose)

        # #######
        # ** this isn't working for the outer 1 loop

        if not max_sim:
            new_actual_loss_mat, new_actual_loss = init_loss(
                dataset, func_name, other_dset, max_sim)
            # real_delta = new_actual_loss - actual_loss

            fprint("Supposed loss: %.6e" % loss)
            fprint("Real loss: %.6e" % new_actual_loss)

        # ########

        fprint(f"Finished iteration {i+1} of {num_temps}.", verbose)
        fprint(f"Current loss is %.6e." % loss, verbose)

    return dataset, other_dset, loss_mat, loss


def dual_mc(dsets,
            func_names,
            num_sweeps_list,
            temp_funcs):

    def update_other(other_dset, dset):

        func_name = func_names[0]
        num_sweeps = num_sweeps_list[0]
        temp_func = temp_funcs[0]
        # we want to maximize the similarity between 0s and 1s
        max_sim = True
        # here the main dataset is `other_dset`, i.e. the dataset
        # of 0's.
        dset_1 = dset
        dset_0 = other_dset

        dset_0, _, _, _ = run_mc(dataset=dset_0,
                                 func_name=func_name,
                                 num_sweeps=num_sweeps,
                                 temp_func=temp_func,
                                 max_sim=max_sim,
                                 other_dset=dset_1,
                                 verbose=False)

        return dset_0

    func_name = func_names[1]
    num_sweeps = num_sweeps_list[1]
    temp_func = temp_funcs[1]
    # we want to minimize the similarity between 0s and 1s
    max_sim = False

    import pdb
    # pdb.set_trace()

    final_dset, other_dset, loss_mat, loss = run_mc(
        dataset=dsets[1],
        func_name=func_name,
        num_sweeps=num_sweeps,
        temp_func=temp_func,
        max_sim=max_sim,
        other_dset=dsets[0],
        update_other=update_other,
        verbose=True)

    return final_dset, other_dset, loss_mat, loss


def to_single_prop(nff_dset, prop, val):
    idx = [i for i, value in enumerate(nff_dset.props[prop])
           if value == val]
    new_props = {}
    for key, val in nff_dset.props.items():
        if type(val) is list:
            new_props[key] = [val[i] for i in idx]
        else:
            new_props[key] = val[idx]
    nff_dset.props = new_props

    return nff_dset


def dsets_from_smiles(nff_dset, smiles_dic):
    dsets = []

    for i in range(2):
        new_dic = {smiles: val for smiles, val in smiles_dic.items()
                   if val == i}
        idx = [i for i, smiles in enumerate(nff_dset.props["smiles"])
               if smiles in new_dic]
        new_props = {}
        for key, val in nff_dset.props.items():
            if type(val) is list:
                new_props[key] = [val[i] for i in idx]
            else:
                new_props[key] = val[idx]
        new_dset = copy.deepcopy(nff_dset)
        new_dset.props = new_props

        dsets.append(new_dset)

    return dsets


def get_dsets(nff_dset_path,
              conf_dset_path,
              smiles_path,
              seed=None):

    datasets = []

    new_paths = [conf_dset_path.replace(".pth.tar", f"_{i}.pth.tar")
                 for i in range(2)]

    if all([os.path.isfile(path) for path in new_paths]):
        for new_path in new_paths:
            fprint(f"Loading conformer dataset from {new_path}...")
            dataset = ConfDataset.from_file(new_path)
            datasets.append(dataset)

    elif os.path.isfile(nff_dset_path):

        fprint(f"Loading NFF dataset from {nff_dset_path}...")
        nff_dset = NffDataset.from_file(nff_dset_path)

        fprint(f"Dataset has size {len(nff_dset)}")
        with open(smiles_path, "r") as f:
            smiles_dic = json.load(f)

        nff_dsets = dsets_from_smiles(nff_dset, smiles_dic)

        # nff_dset = to_single_prop(nff_dset, prop, val)

        for i in range(2):

            nff_dset = nff_dsets[i]
            new_path = new_paths[i]

            fprint(f"Dataset {i+1} has size {len(nff_dset)}")
            dataset = ConfDataset(nff_dset, seed)
            fprint(f"Saving conformer dataset {i+1} to {new_path}...")

            dataset.save(new_path)
            datasets.append(dataset)

    else:
        raise FileNotFoundError

    fprint("Done!")

    for dataset in datasets:
        dataset.randomize_conf_fps(seed)

    return datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg_path", type=str,
                        help="Path to JSON that overrides args")
    parser.add_argument("--temp_dics", type=str,
                        help="Dictionary for temperature scheduler",
                        nargs="+")
    parser.add_argument("--num_sweeps", type=list,
                        help="Number of sweeps per temperature")
    parser.add_argument("--summary_path", type=str,
                        help="Where to save summary")
    parser.add_argument("--num_trials", type=int,
                        help="Number of repeats with different seeds")
    parser.add_argument("--nff_path", type=str,
                        help="Path to NFF dataset", default=None,
                        dest="nff_dset_path")
    parser.add_argument("--conf_path", type=str,
                        help="Path to NFF conformer dataset", default=None,
                        dest="conf_dset_path")
    parser.add_argument("--func_names", type=str,
                        help="Name of similarity function",
                        default="cosine_similarity",
                        nargs="+")
    # parser.add_argument("--prop", type=str,
    #                     help="Property to cluster",
    #                     default=PROP)
    # parser.add_argument("--prop_val", type=int,
    #                     help="Desired value of property to cluster",
    #                     default=1)
    parser.add_argument("--smiles_path", type=str,
                        help=("Path to SMILES strings you want to "
                              "use in clustering"),
                        default=SMILES_PATH)

    args = parser.parse_args()

    if args.arg_path is not None:
        with open(args.arg_path, "r") as f:
            arg_dic = json.load(f)
    else:
        arg_dic = args.__dict__
        for key in ["temp_dics"]:
            arg_dic[key] = [json.loads([i]) for i in arg_dic[key]]

    return arg_dic


def save(final_dsets, loss_mat, loss, arg_dic):

    for i, dataset in enumerate(final_dsets):
        dset_path = arg_dic["conf_dset_path"].replace(".pth.tar",
                                                      f"_{i}_convgd.pth.tar")

        fprint(f"Saving new dataset {i} to {dset_path}...")
        dataset.save(dset_path)

    summary_path = arg_dic["summary_path"]
    summ_dic = {"loss_mat": loss_mat, "loss": loss}
    fprint(f"Saving summary to {summary_path}...")
    torch.save(summ_dic, summary_path)

    fprint("Done!")


def average_dsets(dsets):

    att_weight_list = [dset.props["att_weights"] for dset in dsets]
    dset_len = len(dsets[0])
    all_avg_weights = []
    final_dset = copy.deepcopy(dsets[0])

    for spec_idx in range(dset_len):

        avg_weights = torch.stack([lst[spec_idx] for lst in att_weight_list]
                                  ).mean(0)
        conf_idx = avg_weights.argmax().item()

        final_dset.flip_fp(spec_idx, conf_idx)
        all_avg_weights.append(avg_weights)

    final_dset.props["att_weights"] = all_avg_weights

    return final_dset


# def main():

#     arg_dic = parse_args()

#     dataset = get_dset(nff_dset_path=arg_dic["nff_dset_path"],
#                        conf_dset_path=arg_dic["conf_dset_path"],
#                        smiles_path=arg_dic["smiles_path"],
#                        seed=0)

#     temp_func = make_temp_func(temp_dic=arg_dic["temp_dic"],
#                                dset_len=len(dataset))

#     try:

#         dsets = []
#         num_trials = arg_dic["num_trials"]
#         for seed in range(num_trials):
#             print(f"Starting trial {seed + 1} of {num_trials}")
#             dataset.randomize_conf_fps(seed)

#             # fprint(dataset.props["conf_idx"])
#             # fprint(dataset.props["single_fps"][0])

#             dataset, loss_mat, loss = run_mc(dataset=dataset,
#                                              func_name=arg_dic["func_name"],
#                                              num_sweeps=arg_dic["num_sweeps"],
#                                              temp_func=temp_func)
#             dsets.append(copy.deepcopy(dataset))
#             print(f"Completed trial {seed + 1} of {num_trials}.")

#         dataset = average_dsets(dsets)
#         save(dataset, loss_mat, loss, arg_dic)

#     except Exception as e:
#         print(e)
#         pdb.post_mortem()


def main():

    arg_dic = parse_args()

    dsets = get_dsets(nff_dset_path=arg_dic["nff_dset_path"],
                      conf_dset_path=arg_dic["conf_dset_path"],
                      smiles_path=arg_dic["smiles_path"],
                      seed=0)

    temp_funcs = []
    for i, dset in enumerate(dsets):
        temp_func = make_temp_func(temp_dic=arg_dic["temp_dics"][i],
                                   dset_len=len(dset))
        temp_funcs.append(temp_func)
    func_names = arg_dic["func_names"]
    num_sweeps_list = arg_dic["num_sweeps"]

    try:

        num_trials = arg_dic["num_trials"]
        dset_0 = dsets[0]
        dset_1 = dsets[1]
        dset_list = []

        for seed in range(num_trials):
            print(f"Starting trial {seed + 1} of {num_trials}")
            dset_0.randomize_conf_fps(seed)
            dset_1.randomize_conf_fps(seed)

            dset_1, dset_0, loss_mat, loss = dual_mc(dsets,
                                                     func_names,
                                                     num_sweeps_list,
                                                     temp_funcs)

            dset_list.append([copy.deepcopy(dset_0), copy.deepcopy(dset_1)])
            print(f"Completed trial {seed + 1} of {num_trials}.")

        final_dsets = []
        for i in range(2):
            dsets_i = [dsets[i] for dsets in dset_list]
            dset_i = average_dsets(dsets_i)
            final_dsets.append(dset_i)

        save(final_dsets, loss_mat, loss, arg_dic)

    except Exception as e:
        print(e)
        pdb.post_mortem()

# TO-DO: sample dset_0 from full dataset in batches
