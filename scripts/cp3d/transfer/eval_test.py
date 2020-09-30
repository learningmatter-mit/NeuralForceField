import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.nn import Sigmoid
import pickle
import sys
import copy

from nff.data import Dataset
from nff.train import load_model
from nff.data import collate_dicts
from nff.utils.cuda import batch_to, batch_detach
from nff.data.dataset import concatenate_dict

PROP = "sars_cov_one_cl_protease_active"
METRIC_DIC = {"pr_auc": "maximize",
              "roc_auc": "maximize",
              "loss": "minimize"}


def loss_fn(x, y):
    return torch.Tensor(0)


def save(results, targets, save_path, prop):
    sigmoid = Sigmoid()
    y_true = torch.stack(targets[prop]).numpy()
    probas_pred = sigmoid(torch.cat(results[prop])
                          ).reshape(-1).numpy()
    fps = torch.stack(results["fp"]).numpy()
    all_conf_fps = results["conf_fps"]

    smiles_list = targets["smiles"]
    dic = {}

    for i, smiles in enumerate(smiles_list):

        conf_fps = all_conf_fps[i].numpy()
        dic[smiles] = {"true": y_true[i],
                       "pred": probas_pred[i],
                       "fp": fps[i].reshape(-1),
                       "conf_fps": conf_fps}

    with open(save_path, "wb") as f:
        pickle.dump(dic, f)


def prepare_metric(lines, metric):
    header_items = [i.strip() for i in lines[0].split("|")]
    if metric == "prc_auc":
        metric = "pr_auc"
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


def model_from_metric(model, model_path, metric):

    log_path = os.path.join(model_path, "log_human_read.csv")
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
    check_path = os.path.join(model_path, "checkpoints",
                              f"checkpoint-{best_epoch}.pth.tar")

    state_dict = torch.load(check_path, map_location="cpu"
                            )["model"]
    model.load_state_dict(state_dict)
    model.eval()

    return model


def name_from_metric(metric):
    if metric is None:
        save_name = "test_pred_def_metric.pickle"
    else:
        save_name = f"test_pred_{metric}.pickle"
    return save_name


def fprint(msg):

    print(msg)
    sys.stdout.flush()


def fps_and_pred(model, batch, **kwargs):

    outputs, xyz = model.make_embeddings(batch, xyz=None, **kwargs)
    pooled_fp = model.pool(outputs)
    results = model.readout(pooled_fp)
    results = model.add_grad(batch=batch, results=results, xyz=xyz)

    conf_fps = [i.cpu().detach() for i in outputs["conf_fps_by_smiles"]]
    results.update({"fp": pooled_fp,
                    "conf_fps": conf_fps})

    return results


def evaluate(model,
             loader,
             device,
             **kwargs):

    model.eval()
    model.to(device)

    all_results = []
    all_batches = []

    old_pct = 0

    for i, batch in enumerate(loader):

        new_pct = int((i + 1) / len(loader) * 100 )
        if new_pct - old_pct >= 10:
            fprint("%d%% complete" % new_pct)
            old_pct = new_pct

        batch = batch_to(batch, device)
        results = fps_and_pred(model, batch, **kwargs)

        all_results.append(batch_detach(results))
        all_batches.append(batch_detach(batch))

        del results
        del batch

        # if i == 4:
        #     break

    all_results = concatenate_dict(*all_results)
    all_batches = concatenate_dict(*all_batches)

    return all_results, all_batches


def combine_dsets(dsets):

    new_dset = copy.deepcopy(dsets[0])
    for dset in dsets[1:]:
        for key, val in dset.props.items():
            if type(val) is list:
                new_dset.props[key] += val
            else:
                new_dset.props[key] = torch.cat([new_dset.props[key], val])
    return new_dset


def add_train_val(path):
    dsets = []
    for name in ["train", "val"]:
        print(f"Loading {name} set...")
        new_path = os.path.join(path, "{}.pth.tar".format(name))
        dsets.append(Dataset.from_file(new_path))
    # dataset = combine_dsets(dsets)
    return dsets


def main(path,
         device,
         model_path,
         batch_size,
         prop,
         sub_batch_size,
         all_splits,
         save_path,
         metric=None):

    model = load_model(model_path)
    if metric is not None:
        model = model_from_metric(model=model,
                                  model_path=model_path,
                                  metric=metric)
    fprint("Loading test set...")
    dataset = Dataset.from_file(os.path.join(path, "test.pth.tar"))
    datasets = [dataset]
    dset_names = ["test"]
    if all_splits:
        datasets += add_train_val(path)
        dset_names += ["train", "val"]

    for name, dataset in zip(dset_names, datasets):
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            collate_fn=collate_dicts)

        fprint("Evaluating...")
        results, targets = evaluate(model,
                                    loader,
                                    device=device,
                                    sub_batch_size=sub_batch_size)

        fprint("Saving...")

        save_name = name_from_metric(metric)
        if all_splits:
            save_name = save_name.replace("test_", "")
            save_name = save_name.replace(".pickle",
                                          "_{}.pickle".format(name))
        if save_path is None:
            save_path = path

        pickle_path = os.path.join(save_path, save_name)

        save(results=results,
             targets=targets,
             save_path=pickle_path,
             prop=prop)

    fprint("Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        help="Name of model path")
    parser.add_argument('--path', type=str,
                        help="Name of sub-folder path")
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
                        help="Metric to optimize",
                        default=None)
    parser.add_argument('--all_splits', action='store_true',
                        help="Get results for train, val in addition to test")
    parser.add_argument('--save_path', type=str,
                        help="Path to save pickles")

    args = parser.parse_args()

    try:
        main(path=args.path,
             device=args.gpu,
             model_path=args.model_path,
             batch_size=args.batch_size,
             prop=args.prop,
             sub_batch_size=args.sub_batch_size,
             metric=args.metric,
             all_splits=args.all_splits,
             save_path=args.save_path)
    except Exception as e:
        import pdb
        print(e)
        pdb.post_mortem()
