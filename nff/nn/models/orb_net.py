# %%timeit
# 179 ms ± 5.48 ms

from scipy.linalg import sqrtm
import torch
from rdkit import Chem
import json
import numpy as np
import os

PERIODICTABLE = Chem.GetPeriodicTable()
TEMPLATE = """result := xtb(
  structure(xyz =
  {xyz_str}
)
  print_level=3
  print_basis=true
)
"""

JOB_TEMPLATE = """
entos -f json xtb.inp > xtb.json
"""


def get_result(path):
    with open(path, "r") as f:
        dic = json.load(f)
    result = dic["result"]
    return result

def get_density(result):

    orbs = np.array(result["orbitals"])
    occ_orbs = np.array(result["occupations"]).round().astype(int)
    mask = occ_orbs != 0
    n_occ = len(mask.nonzero()[0])
    n_total = len(orbs)
    reduced_orbs = orbs[:n_total, :n_occ]

    density = 2 * np.matmul(reduced_orbs, reduced_orbs.transpose())

    return density

def get_h(result):
    return np.array(result["h0"])

def get_shells(result):
    basis = result['ao_basis']['__Basis']
    shells = basis['electron_shells']

    return shells

def get_l_info(shells):
    l_arr = np.array([shell['angular_momentum'] for shell in shells])
    l_max = max(l_arr)
    max_size = 2 * l_max + 1

    return l_arr, l_max, max_size

def init_pad(l_arr, max_size):
    # exclude l = 0
    num_blocks = l_arr.nonzero()[0].shape[0]
    pad_mat_dim = max_size * num_blocks
    padded = np.zeros((num_blocks, max_size, max_size))
    l_max = l_arr.max()
    pad_amounts = l_max - l_arr

    return padded, num_blocks, pad_amounts

def fill_pad(shells, density, padded):

    start_idx = 0
    end_idx = 0
    counter = 0
    
    for shell in shells:
        l = shell['angular_momentum']
        if l == 0:
            continue
            
        num_sub = 2 * l + 1
        end_idx += num_sub 

        sub_block = density[start_idx: end_idx,
                            start_idx: end_idx]
        padded[counter, :2 * l + 1, :2* l + 1] = sub_block
        
        counter += 1
        start_idx += num_sub

    return padded
    
def to_y(padded, l_arr, pad_amounts):
    eigs = np.linalg.eigh(padded)
    num_basis = (2 * l_arr + 1).sum()
    y = np.zeros((num_basis, num_basis))
    pad_counter = 0
    y_counter = 0

    for l, pad_amount in zip(l_arr, pad_amounts):
        num = 2 * l + 1
        if l == 0:
            diag_block = np.array([[1]])
        else:
            # this might be wrong for the padded arrays
            diag_block = eigs[1][pad_counter][pad_amount:,
                                              pad_amount:]
        y[y_counter: y_counter + num,
          y_counter: y_counter + num] = diag_block

        y_counter += num
        if l != 0:
            pad_counter += 1
    return y

def make_y(result, shells, density):

    l_arr, l_max, max_size = get_l_info(shells)
    padded, num_blocks, pad_amounts = init_pad(l_arr, max_size)
    paddded = fill_pad(shells, density, padded)
    y = to_y(padded, l_arr, pad_amounts)
    
    return y

def make_y_prime(y, result):
    
    S = np.array(result['overlap'])
    y_prime = np.matmul(y, sqrtm(S))
    return y_prime

def get_starts(shells):
    starts = [0]
    old_idx = None
    n_orb = 0
    for shell in shells:
        idx = shell["center_index"]
        if old_idx != idx:
            starts.append(starts[-1])
        l = shell["angular_momentum"]
        num = 2 * l + 1
        n_orb += num
        starts[-1] += num
        old_idx = idx
        
    n_at = len(starts) - 1    
    return starts, n_orb, n_at
    
def make_q_and_lam(y_prime, n_orb, n_at, starts):
    
    lam = np.zeros((n_orb, n_at))
    for a in range(len(starts) -1):
        start = starts[a]
        end = starts[a + 1]
        lam[start:end, a] = 1
    
    Q = np.einsum('up,uq,ua->pqa', y_prime, y_prime, lam)
    return Q, lam

def get_xyz(result):
    xyz = np.array(result['_input']['structure']['xyz'])[:, 1:].astype("float")
    xyz = torch.Tensor(xyz)
    xyz.requires_grad = True
    
    return xyz

def get_d(xyz):
    n = xyz.size(0)
    d = (xyz.expand(n, n, 3) - xyz.expand(n, n, 3).transpose(0, 1)
            ).norm(dim=2)
    return d

def make_gamma_ab(eta, gamma, d):
    gamma_ab = 1 / (d ** gamma + eta ** (-gamma)) ** (1 / gamma)
    return gamma_ab


def make_j(q, gamma_j):
    if not isinstance(q, torch.Tensor):
        q = torch.Tensor(q)
    result = torch.einsum('ppa,qqb,ab->pq', q, q, gamma_j)
    return result
    
def make_k(q, gamma_k):
    result = torch.einsum('pqa,pqb,ab->pq', q, q, gamma_k)
    return result

def make_node_feats(lam, n_at, ops):
    all_node_feats = []
    
    for op in ops:
        diag_op = torch.diag(op)
        max_size = max([i.nonzero().reshape(-1).shape[0] for i in lam.transpose(0, 1)])
        node_feats = torch.zeros((n_at, max_size))

        for a, idx in enumerate(lam.transpose(0, 1)):
            these_node_feats = diag_op[idx.to(torch.bool)]
            size = these_node_feats.shape[0]
            these_node_feats = torch.cat([these_node_feats, torch.zeros(max_size - size)])
            node_feats[a] = these_node_feats
            
        all_node_feats.append(node_feats)
    
    all_node_feats = torch.cat(all_node_feats, dim=-1)
    return all_node_feats, max_size


def make_edge_feats(n_at, max_size, bool_lam, ops):
    all_edges = []
    for op in ops:
        od_op = op - torch.diagflat(torch.diag(op))
        edges = torch.zeros(n_at, n_at, max_size * max_size)
        
        for a in range(n_at):
            for b in range(n_at):
                if a == b:
                    continue
                edge = od_op[bool_lam[:, a], :][:, bool_lam[:, b]].reshape(-1)
                edges[a, b, :edge.shape[0]] = edge
        all_edges.append(edges)
    all_edges = torch.cat(all_edges, dim=-1)
    
    return all_edges

def test():
    path = "/home/saxelrod/entos_test/xtb.json"
    
    result = get_result(path)
    shells = get_shells(result)
    density = get_density(result)
    h = get_h(result)
    
    y = make_y(result, shells, density)

    y_prime = make_y_prime(y, result)
    starts, n_orb, n_at = get_starts(shells)
    Q, lam = make_q_and_lam(y_prime, n_orb, n_at, starts)

    eta = 0.1
    gamma_j = 4
    gamma_k = 10

    xyz = get_xyz(result)
    d = get_d(xyz)
    
    Q = torch.Tensor(Q)
    lam = torch.Tensor(lam)
    density = torch.Tensor(density)
    h = torch.Tensor(h)
    
    gamma_j_ab = make_gamma_ab(eta, gamma_j, d)
    gamma_k_ab = make_gamma_ab(eta, gamma_k, d)
    
    j = make_j(Q, gamma_j_ab)
    k = make_k(Q, gamma_k_ab)
    
    ops = [j, k, density, h]
    node_feats, max_size = make_node_feats(lam, n_at, ops=ops)
    edge_feats = make_edge_feats(n_at, max_size, bool_lam=lam.to(torch.bool),
                                 ops=ops)
    
    return node_feats, edge_feats, Q, lam, j, k


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
            xyz_str+=",\n"
    return xyz_str

def render_template(xyz_str, job_dir):
    text = TEMPLATE.format(xyz_str=xyz_str)
    path = os.path.join(job_dir, "xtb.inp")
    with open(path, "w") as f:
        f.write(text)

def make_inp_file(xyz, job_dir):
    xyz_str = make_xyz_str(xyz)
    render_template(xyz_str, job_dir)

def make_bash_file(job_dir):
    path = os.path.join(job_dir, "job.sh")
    text = JOB_TEMPLATE
    with open(path, "w") as f:
        f.write(text)

def run(xyz, job_dir):
    make_inp_file(xyz, job_dir)
    make_bash_file(job_dir)
    cmd = f"cd {job_dir} && bash job.sh"
    os.system(cmd)

