from nff.io import make_cp_features
from nff.data import Dataset


def make_smiles_file(info):
    d_path = info["dataset_path"]
    target = info["target_name"]

    dataset = Dataset.from_file(d_path)
    props = dataset.props

    text = "smiles,activity\n"
    text += "\n".join(",".join([smiles, str(act.item())])
                      for smiles, act in zip(props["smiles"], props[target]))

    cp_dic = info["set_params"]["chemprop"]
    load_dic = cp_dic["load_dics"][0]
    smiles_path = load_dic["smiles_path"]

    print("Writing smiles/activity to {}".format(smiles_path))
    with open(smiles_path, "w") as f:
        f.write(text)


def make_all_cp_feats(info):
    cp_dic = info["set_params"]["chemprop"]
    load_dic = cp_dic["load_dics"][0]

    smiles_path = load_dic["smiles_path"]
    features_paths = load_dic["features_path"]

    feat_dics = cp_dic["extra_features"]
    feat_types = [dic["name"] for dic in feat_dics]
    cp_dir = cp_dic["cp_dir"]

    assert len(feat_types) == len(features_paths)

    for feat_path, feat_type in zip(features_paths, feat_types):
        msg = ("Creating {} chemprop features "
               "from {} and saving to {}".format(
               	feat_type, smiles_path, feat_path))
        print(msg)
        make_cp_features(cp_dir=cp_dir,
                         smiles_path=smiles_path,
                         save_path=feat_path,
                         feat_type=feat_type)


def create_features(info):
    if "chemprop" in info["set_params"]:
        make_smiles_file(info)
        make_all_cp_feats(info)

def preprocess(info):
	create_features(info)

