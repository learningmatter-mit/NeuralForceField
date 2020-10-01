import sys
from tqdm import tqdm
import json
import subprocess

def tqdm_enum(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1


def parse_args(parser, config_flag="config_file"):
    args = parser.parse_args()

    config_path = getattr(args, config_flag, None)
    if config_path is not None:
        with open(config_path, "r") as f:
            config_args = json.load(f)
        for key, val in config_args.items():
            if hasattr(args, key):
                setattr(args, key, val)
    return args


def fprint(msg):
    print(msg)
    sys.stdout.flush()

def bash_command(cmd):
    """ Run a command from the command line using subprocess.
    Args:
        cmd (str): command
    Returns:
        None
    """

    return subprocess.Popen(cmd, shell=True, executable='/bin/bash')