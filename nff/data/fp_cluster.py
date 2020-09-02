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
from tqdm import tqdm
import multiprocessing


from nff.data import Dataset as NffDataset

SIM_FUNCS = {"cosine_similarity": F.cosine_similarity}
PROP = "sars_cov_one_cl_protease_active"
SMILES_PATH = "/home/saxelrod/rgb_nfs/GEOM_DATA_ROUND_2/mc_smiles.json"


class ConfDataset(TorchDataset):

    def __init__(self, nff_dset=None, props=None, seed=None):
        if nff_dset is not None:
            self.props = self.make_props(nff_dset, seed)
        elif props is not None:
            self.props = props

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
        smiles = nff_dset.props["smiles"]

        props = {"num_confs": num_confs,
                 "conf_idx": conf_idx,
                 "att_weights": att_weights,
                 "all_fps": all_fps,
                 "single_fps": single_fps,
                 "smiles": smiles}

        return props

    def flip_fp(self, spec_idx, conf_idx, reset_att=True):

        new_fp = self.props["all_fps"][spec_idx][conf_idx]
        self.props["single_fps"][spec_idx] = new_fp
        self.props["conf_idx"][spec_idx] = conf_idx

        self.props["att_weights"][spec_idx] *= 0
        self.props["att_weights"][spec_idx][conf_idx] = 1

    def compare(self, spec_idx, func_name, other_dset):

        other_fps = torch.stack(other_dset.props["single_fps"])
        length = len(other_dset)
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


def dset_from_idx(dset, idx):
    new_props = {}
    for key, val in dset.props.items():
        if type(val) is list:
            new_props[key] = [val[i] for i in idx]
        else:
            new_props[key] = val[idx]

    new_dset = ConfDataset(props=new_props)
    new_dset.props = new_props

    return new_dset


def sample(other_dset, batch_size):

    length = len(other_dset)
    pop = list(range(length - 1))
    sample_idx = random.sample(pop, k=batch_size)
    dset = dset_from_idx(other_dset, sample_idx)

    return dset


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


def get_outer_loss(dset_0,
                   func_name,
                   dset_1):

    sim_other_mat = torch.zeros((len(dset_1), len(dset_0)))
    sim_self_mat = torch.zeros((len(dset_1), len(dset_1)))

    # vectorize

    for spec_idx in range(len(dset_1)):

        sim_other = dset_1.compare(spec_idx, func_name, dset_0)
        sim_other_mat[spec_idx, :] = sim_other

        sim_self = dset_1.compare(spec_idx, func_name, dset_1)
        sim_self_mat[spec_idx, :] = sim_self

    loss = (sim_other_mat.mean() - sim_self_mat.mean()).item()

    return loss


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


def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1


def apply_tdqm(lst, track_prog):
    if track_prog:
        return (tqdm_enumerate(lst))
    else:
        return enumerate(lst)


def inner_opt(dset_0,
              dset_1,
              func_name,
              verbose=False,
              debug=False):

    all_fps_0 = dset_0.props["all_fps"]
    num_fps_0 = [len(i) for i in all_fps_0]
    cat_fps_0 = torch.cat(all_fps_0)  # shape 15989 x 300
    # repeat it as [conf_0, conf_0, ..., conf_1, conf_1, ...]

    len_1 = len(dset_1)
    reshape_fps_0 = cat_fps_0.reshape(-1, cat_fps_0.shape[1], 1)
    repeat_fps_0 = torch.repeat_interleave(
        reshape_fps_0, len_1, 2).transpose(1, 2)

    best_fps_1 = torch.stack(dset_1.props["single_fps"])
    len_0 = cat_fps_0.shape[0]
    repeat_fps_1 = best_fps_1.repeat(len_0, 1, 1)

    sim_func = SIM_FUNCS[func_name]
    sim = sim_func(repeat_fps_0, repeat_fps_1, dim=2).mean(1)

    if debug:

        fprint(("Checking that the inner opt is working."))
        check_sim = []
        for spec_idx in range(len(dset_0)):
            sims = []
            for fp in dset_0.props["all_fps"][spec_idx]:
                sim_mean = 0
                for other_fp in dset_1.props["single_fps"]:
                    sim_mean += sim_func(fp, other_fp, dim=0)
                sim_mean /= len(dset_1)
                sims.append(sim_mean)
            check_sim += sims
        check_sim = torch.Tensor(check_sim)

        print("Difference is %.2e" % abs((check_sim - sim).mean()))
        sys.exit()

    split = torch.split(sim, num_fps_0)
    conf_idx = torch.stack([i.argmax() for i in split])

    for spec_idx, conf_idx in enumerate(conf_idx):
        dset_0.flip_fp(spec_idx, conf_idx)

    return dset_0


def get_sample_0(dset_0, batch_size_0):
    if batch_size_0 is not None:
        sampled_0 = sample(dset_0, batch_size_0)
    else:
        sampled_0 = dset_0
    return sampled_0


def outer_mc(dset_0,
             dset_1,
             func_name,
             num_sweeps,
             temp_func,
             update_0_fn,
             verbose=False,
             batch_size_0=None):

    sample_0 = get_sample_0(dset_0, batch_size_0)
    loss = get_outer_loss(dset_0=sample_0,
                          func_name=func_name,
                          dset_1=dset_1)

    temps = temp_func()
    num_temps = len(temps)

    fprint(f"Starting loss without equilibration is %.6e." % loss, verbose)

    sample_0 = update_0_fn(sample_0, dset_1)
    loss = get_outer_loss(dset_0=sample_0,
                          func_name=func_name,
                          dset_1=dset_1)

    fprint(f"Starting loss after equilibration is %.6e." % loss, verbose)

    tracked_loss = []

    for i, temp in enumerate(temps):

        fprint((f"Starting iteration {i+1} of {num_temps} at "
                "temperature %.3e... " % temp), verbose)

        num_iters = int(len(dset_1) * num_sweeps)

        for it in apply_tdqm(range(num_iters), verbose):

            spec_idx, conf_idx = choose_flip(dset_1)
            old_conf_idx = dset_1.props["conf_idx"][spec_idx]
            dset_1.flip_fp(spec_idx, conf_idx)

            sample_0 = get_sample_0(dset_0, batch_size_0)
            sample_0 = update_0_fn(sample_0, dset_1)
            new_loss = get_outer_loss(dset_0=sample_0,
                                      func_name=func_name,
                                      dset_1=dset_1)

            # delta_loss = new_loss - loss
            # print("Change in loss is %.2e" % delta_loss)

            delta_loss = torch.Tensor([new_loss - loss]).squeeze()
            criterion = get_criterion(delta_loss, temp)

            if criterion:
                loss = new_loss

            else:

                dset_1.flip_fp(spec_idx, old_conf_idx)

            tracked_loss.append(loss)

        fprint(f"Finished iteration {i+1} of {num_temps}.", verbose)
        fprint(f"Current loss is %.6e." % loss, verbose)

    return dset_1, tracked_loss


def dual_mc(dset_0,
            dset_1,
            inner_func_name,
            outer_func_name,
            outer_num_sweeps,
            outer_temp_dic,
            batch_size_0,
            debug=False):

    def update_0_fn(dset_0, dset_1):

        dset_0 = inner_opt(dset_0=dset_0,
                           dset_1=dset_1,
                           func_name=inner_func_name,
                           verbose=False,
                           debug=debug)

        return dset_0

    outer_temp_func = make_temp_func(temp_dic=outer_temp_dic,
                                     dset_len=len(dset_1))

    dset_1, tracked_loss = outer_mc(dset_0=dset_0,
                                    dset_1=dset_1,
                                    func_name=outer_func_name,
                                    num_sweeps=outer_num_sweeps,
                                    temp_func=outer_temp_func,
                                    update_0_fn=update_0_fn,
                                    verbose=True,
                                    batch_size_0=batch_size_0)

    return dset_1, tracked_loss


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

        new_dset = NffDataset(new_props)
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

        for i in range(2):

            nff_dset = nff_dsets[i]
            new_path = new_paths[i]

            fprint(f"Dataset {i} has size {len(nff_dset)}")
            dataset = ConfDataset(nff_dset, seed)
            fprint(f"Saving conformer dataset {i} to {new_path}...")

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
    parser.add_argument("--outer_temp_dic", type=str,
                        help="Dictionary for outer temperature scheduler",
                        nargs="+")
    parser.add_argument("--outer_num_sweeps", type=list,
                        help="Outer number of sweeps per temperature")
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
    parser.add_argument("--inner_func_name", type=str,
                        help="Name of inner similarity function",
                        default="cosine_similarity",
                        nargs="+")
    parser.add_argument("--outer_func_name", type=str,
                        help="Name of outer similarity function",
                        default="cosine_similarity",
                        nargs="+")
    parser.add_argument("--smiles_path", type=str,
                        help=("Path to SMILES strings you want to "
                              "use in clustering"),
                        default=SMILES_PATH)
    parser.add_argument("--batch_size_0", type=int,
                        help="Batch size for the 0 dataset")

    args = parser.parse_args()

    if args.arg_path is not None:
        with open(args.arg_path, "r") as f:
            arg_dic = json.load(f)
    else:
        arg_dic = args.__dict__
        for key in ["outer_temp_dic"]:
            arg_dic[key] = json.loads(arg_dic[key])

    return arg_dic


def save(dset_1, tracked_loss, arg_dic):

    dset_path = arg_dic["conf_dset_path"].replace(".pth.tar",
                                                  "_1_convgd.pth.tar")

    fprint(f"Saving new dataset 1 to {dset_path}...")
    dset_1.save(dset_path)

    summary_path = arg_dic["summary_path"]
    summ_dic = {"tracked_loss": tracked_loss}
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


def main(debug=False):

    arg_dic = parse_args()

    dsets = get_dsets(nff_dset_path=arg_dic["nff_dset_path"],
                      conf_dset_path=arg_dic["conf_dset_path"],
                      smiles_path=arg_dic["smiles_path"],
                      seed=0)

    # make sure torch uses available cores for computations
    num_cpus = multiprocessing.cpu_count()
    torch.set_num_threads(num_cpus)

    try:

        num_trials = arg_dic["num_trials"]
        dset_0 = dsets[0]
        dset_1 = dsets[1]
        dset_1_list = []

        for seed in range(num_trials):

            fprint(f"Starting trial {seed + 1} of {num_trials}")
            dset_0.randomize_conf_fps(seed)
            dset_1.randomize_conf_fps(seed)

            dset_1, tracked_loss = dual_mc(
                dset_0=dset_0,
                dset_1=dset_1,
                inner_func_name=arg_dic["inner_func_name"],
                outer_func_name=arg_dic["outer_func_name"],
                outer_num_sweeps=arg_dic["outer_num_sweeps"],
                outer_temp_dic=arg_dic["outer_temp_dic"],
                batch_size_0=arg_dic["batch_size_0"],
                debug=debug)

            dset_1_list.append(copy.deepcopy(dset_1))
            fprint(f"Completed trial {seed + 1} of {num_trials}.")

        dset_1 = average_dsets(dset_1_list)
        save(dset_1, tracked_loss, arg_dic)

    except Exception as e:
        fprint(e)
        pdb.post_mortem()
