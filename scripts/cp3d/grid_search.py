import os
import numpy as np
import json

MODEL_PATH = "/home/gridsan/saxelrod/models/1072"
JOB_PATH = "/home/gridsan/saxelrod/final_cp3d_train_25k"
MAX_DROP = 1.0
NUM_SAMPLES = 11


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


def update_info(job_path, dropout):

    info_file = os.path.join(job_path, "job_info.json")
    with open(info_file, "r") as f:
        info = json.load(f)
    info["schnet_dropout"] = dropout
    info["details"]["schnet_dropout"] = dropout

#    readout = info["details"]["readoutdict"]
#    layer_dics = readout["sars_cov_one_cl_protease_active"]
#    for layer_dic in layer_dics:
#        if layer_dic["name"] == "Dropout":
#            layer_dic["param"]["p"] = dropout
#    info["details"]["readoutdict"] = {
#        "sars_cov_one_cl_protease_active": layer_dics}
    with open(info_file, "w") as f:
        json.dump(info, f, indent=4, sort_keys=True)


def main(job_path=JOB_PATH,
         model_path=MODEL_PATH,
         max_drop=MAX_DROP,
         num_samples=NUM_SAMPLES):

    dropouts = np.linspace(0, max_drop, num_samples)
    # dropouts = np.linspace(0.1, max_drop, num_samples)
    auc_dic = {}
    auc_path = os.path.join(job_path, "auc.json")
    for dropout in dropouts:
        clean_up(model_path=model_path)
        update_info(job_path=job_path, dropout=dropout)
        best_auc = run(job_path=job_path,
                       model_path=model_path)
        auc_dic[dropout] = best_auc

        with open(auc_path, "w") as f:
            json.dump(auc_dic, f, indent=4, sort_keys=True)

if __name__ == "__main__":
    main()

