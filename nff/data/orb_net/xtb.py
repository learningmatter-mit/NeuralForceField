"""
Featurize with xtb
"""

# 325 += 29 ms for azobenzene

from rdkit import Chem
import json
import numpy as np
import os
import subprocess

PERIODICTABLE = Chem.GetPeriodicTable()
TEMPLATE = """result := xtb(
  structure(xyz =
  {xyz_str}
)
  print_level=3
  print_basis=true
  no_fail={no_fail}
  ! diis=cdiis
  model=gfn0 ! gfn1 has convergence problems
  ! max_iter=400
)
"""


def make_xyz_str(xyz):
    xyz_str = "["
    for i, quad in enumerate(xyz):
        z = int(quad[0])
        element = str(PERIODICTABLE.GetElementSymbol(z))
        coord_str = ", ".join(np.array(quad[1:]).astype("str"))
        xyz_str += f"[{element}, {coord_str}]"
        if i == len(xyz) - 1:
            xyz_str += "]"
        else:
            xyz_str += ",\n"

    return xyz_str


def render_template(xyz_str,
                    job_dir,
                    no_fail=False):
    text = TEMPLATE.format(xyz_str=xyz_str,
                           no_fail=json.dumps(no_fail))
    path = os.path.join(job_dir, "xtb.inp")
    with open(path, "w") as f:
        f.write(text)


def make_inp_file(xyz, job_dir):
    xyz_str = make_xyz_str(xyz)
    render_template(xyz_str, job_dir)


def make_bash_file(job_dir):
    path = os.path.join(job_dir, "job.sh")
    text = "entos -f json xtb.inp > xtb.json"
    with open(path, "w") as f:
        f.write(text)


def run_xtb(xyz, job_dir):
    make_inp_file(xyz, job_dir)
    make_bash_file(job_dir)
    cmd = f"cd {job_dir} && bash job.sh"
    p = subprocess.Popen([cmd],
                         shell=True,
                         stdin=None,
                         stdout=None,
                         stderr=None,
                         close_fds=True)
    p.wait()


if __name__ == "__main__":
    xyz_path = ("/home/saxelrod/Repo/projects/ax_autopology/"
                "NeuralForceField/orb_net/xyz_test.json")
    with open(xyz_path, "r") as f:
        xyz_test = json.load(f)
    job_dir = ('/home/saxelrod/Repo/projects/ax_autopology/'
               'NeuralForceField/orb_net')
    run_xtb(xyz_test, job_dir)
