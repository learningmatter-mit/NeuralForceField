import os
import numpy as np
import json

MODEL_PATH = "/home/gridsan/saxelrod/models/1072"
JOB_PATH = "/home/gridsan/saxelrod/final_cp3d_train_25k"
NUM_SAMPLES = 10


def clean_up(model_path):
    cmds = ["rm */*.pickle",
            "rm -rf check*",
            "rm */train_len",
            "rm *.csv",
            "rm */*.csv",
            "rm -r best_model",
            "rm */*epoch*"]

    for cmd in cmds:
        os.system("cd {} && {}".format(model_path, cmd))


def run(job_path, model_path):

    cmd = "cd {} && bash job.sh".format(job_path)
    os.system(cmd)

    log_path = os.path.join(model_path, "log_human_read.csv")
    with open(log_path, "r") as f:
        lines = f.readlines()

    aucs = []
    for line in reversed(lines):
        try:
            aucs.append(float(line.split("|")[-2]))
        except:
            continue

    best_auc = max(aucs)
    return best_auc


def update_info(job_path, heads):

    info_file = os.path.join(job_path, "job_info.json")
    with open(info_file, "r") as f:
        info = json.load(f)
    info["details"]["boltzmann_dict"]["num_heads"] = heads
    info["details"]["boltzmann_dict"]["head_pool"] = "concatenate"

    readoutdict = info["details"]["readoutdict"]
    input_layers = info["details"]["input_layers"]
    feat_dim = input_layers[0]["param"]["out_features"]

    for key, lst in readoutdict.items():
        for i, dic in enumerate(lst):
            if "param" in dic and "in_features" in dic.get("param", {}):
                readoutdict[key][i]["param"]["in_features"] = feat_dim * heads
                break
    info["details"]["readoutdict"] = readoutdict

    with open(info_file, "w") as f:
        json.dump(info, f, indent=4, sort_keys=True)


def main(job_path=JOB_PATH,
         model_path=MODEL_PATH,
         num_samples=NUM_SAMPLES):

    head_list = list(range(4, num_samples))

    auc_dic = {}
    auc_path = os.path.join(job_path, "heads_auc.json")
    for heads in head_list:
        clean_up(model_path=model_path)
        update_info(job_path=job_path, heads=heads)
        best_auc = run(job_path=job_path,
                       model_path=model_path)
        auc_dic[heads] = best_auc

        with open(auc_path, "w") as f:
            json.dump(auc_dic, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()

