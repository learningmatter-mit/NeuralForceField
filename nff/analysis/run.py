import sys
sys.path.insert(0, "/home/saxelrod/Repo/projects/master/NeuralForceField")

import os
import copy
import json

from nff.analysis import conf_sims_from_files

NAME = "cov_2_gen"
# model_path = "/home/saxelrod/rgb_nfs/supercloud_backup/models/saved_models/cov_2_cl/schnet_feat/seed_0"
model_path = "/home/saxelrod/rgb_nfs/supercloud_backup/models/saved_models/cov_2_gen/schnet_feat"
rd_path = "/home/saxelrod/rgb_nfs/GEOM_DATA_ROUND_2/rdkit_folder"
summary_path = os.path.join(rd_path, "summary_drugs.json")

def recurs_json(dic):
    for key, val in dic.items():
        if isinstance(val, dict):
            recurs_json(val)
        elif hasattr(val, "tolist"):
            dic[key] = val.tolist()
        elif isinstance(val, list):
            if hasattr(val[0], "tolist"):
                dic[key] = [sub_val.tolist()
                           for sub_val in val]

def to_json(dic):
    new_dic = copy.deepcopy(dic)
    recurs_json(new_dic)
    return new_dic

analysis, bare_dic = conf_sims_from_files(model_path=model_path,
                                                max_samples=5000,
                                                classifier=True,
                                                seed=0,
                                                external_fp_fn="e3fp",
                                                summary_path=summary_path,
                                                rd_path=rd_path,
                                                fp_kwargs={"bits": 256})


bare_json = to_json(bare_dic)

with open(f"{NAME}_analysis.json", "w") as f:
    json.dump(analysis, f, indent=4, sort_keys=True)
with open(f"{NAME}_bare.json", "w") as f:
    json.dump(bare_json, f, indent=4, sort_keys=True)
