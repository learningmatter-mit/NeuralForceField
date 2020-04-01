import subprocess
import os


def make_cp_features(cp_dir, smiles_path, save_path, feat_type):
    """
    Call chemprop to make external features
    """

    script_path = os.path.join(cp_dir, "scripts/save_features.py")
    feat_cmds = ["python", script_path, "--data_path",
                 smiles_path, "--save_path", save_path,
                 "--features_generator", feat_type, "--restart"]
    subprocess.check_output(feat_cmds)
