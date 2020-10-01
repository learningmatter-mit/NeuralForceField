import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.nn import Sigmoid
import pickle
import sys
from tqdm import tqdm

from nff.data import Dataset
from nff.train import load_model
from nff.data import collate_dicts
from nff.utils.cuda import batch_to, batch_detach
from nff.data.dataset import concatenate_dict
from nff.utils import tqdm_enum, parse_args

PROP = "sars_cov_one_cl_protease_active"
METRIC_DIC = {"pr_auc": "maximize",
              "roc_auc": "maximize",
              "loss": "minimize"}


def save(results,
         targets,
         save_path,
         prop,
         add_sigmoid):

    y_true = torch.stack(targets[prop]).numpy()

    if add_sigmoid:
        sigmoid = Sigmoid()
        probas_pred = sigmoid(torch.cat(results[prop])
                              ).reshape(-1).numpy()
    else:
        probas_pred = (torch.cat(results[prop])
                       .reshape(-1).numpy())

    fps = torch.stack(results["fp"]).numpy()
    all_conf_fps = results["conf_fps"]
    learned_weights = results["learned_weights"]
    energy = results["energy"]
    boltz_weights = results["boltz_weights"]

    smiles_list = targets["smiles"]
    dic = {}

    for i, smiles in enumerate(smiles_list):

        conf_fps = all_conf_fps[i].numpy()
        dic[smiles] = {"true": y_true[i],
                       "pred": probas_pred[i],
                       "fp": fps[i].reshape(-1),
                       "conf_fps": conf_fps,
                       "learned_weights": learned_weights[i].numpy(),
                       "energy": energy[i].reshape(-1).numpy(),
                       "boltz_weights": boltz_weights[i].reshape(-1).numpy()}

    with open(save_path, "wb") as f:
        pickle.dump(dic, f)


def prepare_metric(lines, metric):
    header_items = [i.strip() for i in lines[0].split("|")]
    if metric in ["prc_auc", "prc-auc"]:
        metric = "pr_auc"
    elif metric in ["auc", "roc-auc"]:
        metric = "roc_auc"
    if metric == "loss":
        idx = header_items.index("Validation loss")
    else:
        for i, item in enumerate(header_items):
            sub_keys = metric.split("_")
            if all([key.lower() in item.lower()
                    for key in sub_keys]):
                idx = i

    optim = METRIC_DIC[metric]

    if optim == "minimize":
        best_score = float("inf")
    else:
        best_score = -float("inf")

    best_epoch = -1

    return idx, best_score, best_epoch, optim


def model_from_metric(model, model_folder, metric):

    log_path = os.path.join(model_folder, "log_human_read.csv")
    with open(log_path, "r") as f:
        lines = f.readlines()

    idx, best_score, best_epoch, optim = prepare_metric(
        lines=lines,
        metric=metric)

    for line in lines:
        splits = [i.strip() for i in line.split("|")]
        try:
            score = float(splits[idx])
        except ValueError:
            continue
        if any([(optim == "minimize" and score < best_score),
                (optim == "maximize" and score > best_score)]):
            best_score = score
            best_epoch = splits[1]
    check_path = os.path.join(model_folder, "checkpoints",
                              f"checkpoint-{best_epoch}.pth.tar")

    state_dict = torch.load(check_path, map_location="cpu"
                            )["model"]
    model.load_state_dict(state_dict)
    model.eval()

    return model


def fprint(msg):

    print(msg)
    sys.stdout.flush()


def fps_and_pred(model, batch, **kwargs):

    outputs, xyz = model.make_embeddings(batch, xyz=None, **kwargs)
    pooled_fp, learned_weights = model.pool(outputs)
    results = model.readout(pooled_fp)
    results = model.add_grad(batch=batch, results=results, xyz=xyz)

    conf_fps = [i.cpu().detach() for i in outputs["conf_fps_by_smiles"]]
    energy = batch["energy"]
    boltz_weights = batch["weights"]

    results.update({"fp": pooled_fp,
                    "conf_fps": conf_fps,
                    "learned_weights": learned_weights,
                    "energy": energy,
                    "boltz_weights": boltz_weights})
    return results


def evaluate(model,
             loader,
             device,
             **kwargs):

    model.eval()
    model.to(device)

    all_results = []
    all_batches = []

    for i, batch in tqdm_enum(loader):

        batch = batch_to(batch, device)
        results = fps_and_pred(model, batch, **kwargs)

        all_results.append(batch_detach(results))
        all_batches.append(batch_detach(batch))

    all_results = concatenate_dict(*all_results)
    all_batches = concatenate_dict(*all_batches)

    return all_results, all_batches


def get_dsets(full_path, test_only):
    if test_only:
        dset_names = ['test']
    else:
        dset_names = ["train", "val", "test"]

    dsets = []
    for name in tqdm(dset_names):
        new_path = os.path.join(full_path, "{}.pth.tar".format(name))
        dsets.append(Dataset.from_file(new_path))

    return dsets, dset_names


def main(dset_folder,
         gpu,
         model_folder,
         batch_size,
         prop,
         sub_batch_size,
         save_path,
         add_sigmoid,
         metric=None,
         test_only=False,
         **kwargs):

    model = load_model(model_folder)
    if metric is not None:
        model = model_from_metric(model=model,
                                  model_folder=model_folder,
                                  metric=metric)
    datasets, dset_names = get_dsets(dset_folder, test_only)

    for i in tqdm(range(len(dset_names))):
        dataset = datasets[i]
        name = dset_names[i]
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_dicts)

        results, targets = evaluate(model,
                                    loader,
                                    device=gpu,
                                    sub_batch_size=sub_batch_size)

        save_name = f"pred_{metric}_{name}.pickle"
        if save_path is None:
            save_path = dset_folder

        pickle_path = os.path.join(save_path, save_name)

        save(results=results,
             targets=targets,
             save_path=pickle_path,
             prop=prop,
             add_sigmoid=add_sigmoid)

    fprint("Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', type=str,
                        help="Name of model path")
    parser.add_argument('--dset_folder', type=str,
                        help=("Name of the folder with the "
                              "datasets you want to add "
                              "fingerprints to"))
    parser.add_argument('--gpu', type=int,
                        help="Name of gpu to use")
    parser.add_argument('--batch_size', type=int,
                        help="Batch size")
    parser.add_argument('--prop', type=str,
                        help="Property to predict",
                        default=PROP)
    parser.add_argument('--sub_batch_size', type=int,
                        help="Sub batch size",
                        default=7)
    parser.add_argument('--metric', type=str,
                        help=("Select the model with the best validation "
                              "score on this metric. If no metric "
                              "is given, the metric used in the training "
                              "process will be used."),
                        default=None)
    parser.add_argument('--save_path', type=str,
                        help="Path to save pickles")
    parser.add_argument('--test_only', action='store_true',
                        help=("Only evaluate model "
                              "and generate fingerprints for "
                              "the test set"))
    parser.add_argument('--add_sigmoid', action='store_true',
                        help=("Add a sigmoid layer to predictions. "
                              "This should be done if your model is a "
                              "classifier trained with a BCELogits loss. "))
    parser.add_argument('--config_file', type=str,
                        help=("Path to JSON file with arguments. If given, "
                              "any arguments in the file override the command "
                              "line arguments."))
    args = parse_args(parser)

    main(**args.__dict__)
