import json
import numpy as np
import torch
from rdkit import Chem
import os
from scipy.linalg import sqrtm
from tqdm import tqdm
from torch import cos, sin
import copy


from nff.utils.constants import HARDNESS_AU_MAT, BOHR_RADIUS
from nff.data.orb_net.xtb import run_xtb
from nff.data import Dataset
from nff.utils import tqdm_enum

PERIODICTABLE = Chem.GetPeriodicTable()

# j and k overlap exponents
GAMMA_J = 4
GAMMA_K = 10

# cutoffs
CUTOFF_NAMES = ["f", "j", "k", "p", "s", "h"]
CUTOFFS = [8, 1.6, 20, 14, 8, 8]


"""

Percentage kept: {"f": 59.19008875739645, 
                  "j": 11.390532544378699, 
                  "k": 51.60872781065089, 
                  "p": 99.86131656804734, 
                  "s": 60.32729289940828,
                  "h": 59.19008875739645}
                  
 * is this reasonable?

1. Check transformations
2. Are the expressions for J and K in the paper consistent with 
the x transformations in the gradient paper?
3. Why does our density disagree with loaded density?
4. Add D and don't transform it - but I don't think 
   we have the info from the results to make D because we
   don't have the AO dipole matrix.

"""


def get_results(path):
    """
    Load results from qcore xtb calculation.
    Args:
        path (str): path to json file
    Returns:
        results (dict): results dictionary
    """
    with open(path, "r") as f:
        dic = json.load(f)
    results = dic["result"]

    return results


def get_shells(results):
    basis = results['ao_basis']['__Basis']
    shells = basis['electron_shells']

    return shells


def get_orbs(results):
    orbs = np.array(results["orbitals"])
    return orbs


def get_s(results):
    s = np.array(results["overlap"])
    return s


def get_starts(shells):
    l_arr = np.array([shell['angular_momentum'] for shell in shells])
    starts = [0]
    starts_by_at = [0]
    old_center_idx = None

    for l, shell in zip(l_arr, shells):
        num = 2 * l + 1

        center_idx = shell["center_index"]
        if center_idx != old_center_idx:
            starts_by_at.append(starts_by_at[-1])

        starts.append(starts[-1] + num)
        starts_by_at[-1] += num

        old_center_idx = center_idx

    return starts, starts_by_at


def get_p_ao(results):
    """
    ** Why does our density disagree with the loaded density?


    Get density matrix in the AO basis.
    Args:
        results (dict): results dictionary
    Returns:
        p (np.array): dim x dim density matrix,
            where dim is the number of atomic orbitals
            (equal to number of molecular orbitals).
    """

    orbs = np.array(results["orbitals"])
    occ_orbs = np.array(results["occupations"]).round().astype(int)
    mask = occ_orbs != 0
    n_occ = len(mask.nonzero()[0])
    n_total = len(orbs)
    reduced_orbs = orbs[:n_total, :n_occ]

    # density matrix
    p = 2 * np.matmul(reduced_orbs, reduced_orbs.transpose())

    # 99.9% get kept even when using `results["density"]`
    # p = np.array(results["density"])

    return p


def get_x_or_y(shells,
               orbitals,
               p_tilde,
               starts):

    l_arr = np.array([shell['angular_momentum']
                      for shell in shells])
    p_tilde_by_l = {}
    l_idx = {}

    for i, l in enumerate(l_arr):
        if l not in p_tilde_by_l:
            p_tilde_by_l[l] = []
            l_idx[l] = []

        start_idx = starts[i]
        end_idx = starts[i + 1]

        this_p_tilde = p_tilde[start_idx: end_idx,
                               start_idx: end_idx]

        p_tilde_by_l[l].append(this_p_tilde)
        l_idx[l].append([start_idx, end_idx])

    # stack them to vectorize eigenvector calculation
    p_tilde_by_l = {key: np.stack(val) for key, val in
                    p_tilde_by_l.items()}

    # calculate the stacked eigenvectors for each value of l, except l = 0
    eig_blocks = {key: np.linalg.eigh(val)[1] for key, val in
                  p_tilde_by_l.items() if key != 0}

    # 1 for l = 0
    eig_blocks[0] = p_tilde_by_l[0] / p_tilde_by_l[0]

    # fill in transformation matrix
    orb_dim = orbitals.shape[0]
    x = np.zeros((orb_dim, orb_dim))

    for l, all_vecs in eig_blocks.items():
        for i, eigvec in enumerate(all_vecs):
            start, stop = l_idx[l][i]
            x[start: stop, start: stop] = eigvec

    return x


def get_x(shells, orbitals, s, p_ao, starts):

    p_tilde = np.matmul(s, np.matmul(p_ao, s))
    return get_x_or_y(shells, orbitals, p_tilde, starts)


def get_y(shells, orbitals, p_ao, starts):
    return get_x_or_y(shells, orbitals, p_ao, starts)


def transf(op, x):
    x_dag = x.transpose(0, 1)
    op_saao = np.matmul(x_dag, np.matmul(op, x))
    return op_saao


def get_h_saao(results, x):
    h_ao = np.array(results["h0"])
    h_saao = transf(h_ao, x)
    return h_saao


def get_s_saao(s, x):
    s_saao = transf(s, x)
    return s_saao


def get_f_saao(results, x):
    f = np.array(results["fock"])
    f_saao = transf(f, x)
    return f_saao


def get_p_saao(p_ao, x):
    p_saao = transf(p_ao, x)
    return p_saao


def make_q(y_mat,
           s,
           n_at,
           starts_by_at):

    y_prime = np.matmul(y_mat, sqrtm(s))
    n_orb = y_prime.shape[0]
    lam = np.zeros((n_orb, n_at))

    for a in range(len(starts_by_at) - 1):
        start = starts_by_at[a]
        end = starts_by_at[a + 1]
        lam[start:end, a] = 1

    Q = np.einsum('up,uq,ua->pqa', y_prime, y_prime, lam)

    return Q


def qm_from_xtb(nxyz, results):

    shells = get_shells(results)
    p_ao = get_p_ao(results)
    starts, starts_by_at = get_starts(shells)
    orbitals = get_orbs(results)
    s = get_s(results)

    y_mat = get_y(shells, orbitals, p_ao, starts)
    x_mat = get_x(shells, orbitals, s, p_ao, starts)

    q_mat = make_q(y_mat=y_mat,
                   s=s,
                   n_at=len(nxyz),
                   starts_by_at=starts_by_at)

    h_saao = get_h_saao(results, x_mat)
    s_saao = get_s_saao(s, x_mat)
    f_saao = get_f_saao(results, x_mat)
    p_saao = get_p_saao(p_ao, x_mat)

    out_dic = {"p_saao": torch.Tensor(p_saao),
               "f_saao": torch.Tensor(f_saao),
               "h_saao": torch.Tensor(h_saao),
               "s_saao": torch.Tensor(s_saao),
               "q_mat": torch.Tensor(q_mat),
               "starts": starts}

    return out_dic


def get_qm_quants(nxyz, job_dir):

    run_xtb(nxyz.tolist(), job_dir)
    path = os.path.join(job_dir, "xtb.json")
    results = get_results(path)
    out_dic = qm_from_xtb(nxyz, results)

    return out_dic


def qm_to_dset(dset_path, job_dir):

    dset = Dataset.from_file(dset_path)
    props = {}

    for i in tqdm(range(len(dset.props["nxyz"]))):
        nxyz = dset.props["nxyz"][i]
        out_dic = get_qm_quants(nxyz, job_dir)
        for key, val in out_dic.items():
            if key not in props:
                props[key] = []
            props[key].append(val)

    dset.props.update(props)

    return dset


def make_gamma_ab(gamma, d_au_sq, z):

    flat_z = z.reshape(-1)
    n = flat_z.reshape(-1).shape[0]
    eta_A = (HARDNESS_AU_MAT[flat_z]
             .expand(n, n)
             .to(d_au_sq.device))
    eta_B = eta_A.transpose(0, 1)
    eta_avg = 1/2 * (eta_A + eta_B)

    assert gamma % 2 == 0
    gam_by_2 = gamma // 2

    gamma_ab = torch.pow(torch.pow(d_au_sq, gam_by_2)
                         + eta_avg ** (-gamma),
                         -1 / gamma)

    return gamma_ab


def make_j(q, gamma_j):

    results = torch.einsum('ppa,qqb,ab->pq', q, q, gamma_j)

    return results


def make_k(q, gamma_k):

    results = torch.einsum('pqa,pqb,ab->pq', q, q, gamma_k)

    return results


def make_node_feats(ops):
    # will need to keep track of the ordering of the operators
    # to get the appropriate cutoffs
    node_dim = ops[0].shape[0]
    feat_dim = len(ops)
    node_feats = torch.zeros(node_dim, feat_dim)

    for i, op in enumerate(ops):
        diag_op = torch.diag(op)
        node_feats[:, i] = diag_op

    return node_feats


def make_edge_feats(ops):

    node_dim = ops[0].shape[0]
    feat_dim = len(ops)
    edge_feats = torch.zeros(node_dim, node_dim, feat_dim)

    for i, op in enumerate(ops):
        off_diag_op = op - torch.diagflat(torch.diag(op))
        edge_feats[:, :, i] = off_diag_op

    return edge_feats


def normalize_node_feats(props):

    node_feats = props["node_features"]
    num_feats = node_feats[0].shape[-1]
    max_vals = torch.zeros(1, num_feats)
    min_vals = torch.zeros(1, num_feats)

    for i in range(num_feats):
        min_val = min([node_feat[:, i].min()
                       for node_feat in node_feats])
        max_val = max([node_feat[:, i].max()
                       for node_feat in node_feats])
        max_vals[:, i] = max_val
        min_vals[:, i] = min_val

    for i in range(len(props["node_features"])):
        props["node_features"][i] -= min_vals
        props["node_features"][i] /= (max_vals - min_vals)


def normalize_edge_feats(props):

    for i in range(len(props["edge_features"])):

        feats = props["edge_features"][i]
        props["edge_features"][i] = - torch.log(abs(feats))

        # get rid of infinities
        nan_mask = props["edge_features"][i] == float("inf")
        props["edge_features"][i][nan_mask] = 0


def get_dist_z(batch):

    nxyz = batch["nxyz"]
    xyz = nxyz[:, 1:]

    # will probably have to add this in later anyway
    # xyz.requires_grad = True

    n = xyz.shape[0]
    dist_sq = (((xyz.expand(n, n, 3) -
                 xyz.expand(n, n, 3)
                 .transpose(0, 1)) ** 2)
               .sum(-1))

    d_au_sq = dist_sq / BOHR_RADIUS ** 2
    z = nxyz[:, 0].long()

    return d_au_sq, z


def make_ops(batch):

    d_au_sq, z = get_dist_z(batch)
    gamma_j_ab = make_gamma_ab(gamma=GAMMA_J,
                               d_au_sq=d_au_sq,
                               z=z)
    gamma_k_ab = make_gamma_ab(gamma=GAMMA_K,
                               d_au_sq=d_au_sq,
                               z=z)

    j = make_j(batch["q_mat"], gamma_j_ab)
    k = make_k(batch["q_mat"], gamma_k_ab)

    ops = [batch["f_saao"],
           j, k,
           batch["p_saao"],
           batch["s_saao"],
           batch["h_saao"]]

    return ops


def featurize_dset(dset):

    props = {"node_features": [],
             "edge_features": []}

    for i, batch in tqdm_enum(dset):

        ops = make_ops(batch)
        node_feats = make_node_feats(ops)
        edge_feats = make_edge_feats(ops)

        props["node_features"].append(node_feats)
        props["edge_features"].append(edge_feats)

    # normalize
    normalize_node_feats(props)
    normalize_edge_feats(props)

    dset.props.update(props)

    return dset


def generate_neighbors(dset,
                       cutoffs,
                       cutoff_names):

    edge_feats_0 = dset.props["edge_features"][0]
    assert len(cutoffs) == edge_feats_0.shape[-1]

    props = {f"{name}_nbr_list": [] for
             name in cutoff_names}

    for batch in dset:
        edge_feats = batch["edge_features"]
        pcts = [[] for _ in range(len(cutoffs))]
        for i, name in enumerate(cutoff_names):
            mask = edge_feats[:, :, i] <= cutoffs[i]
            nbrs = mask.nonzero()
            props[f"{name}_nbr_list"].append(nbrs)

            pct_kept = (nbrs.shape[0] /
                        edge_feats.shape[0] ** 2) * 100
            pcts[i].append(pct_kept)

    pcts = [np.mean(pct) for pct in pcts]
    print(f"Percentage kept: {pcts}")

    dset.props.update(props)
    return dset


def test_rotate(dset):
    new_dset = copy.deepcopy(dset)
    new_nxyz = []
    for batch in dset:
        xyz = batch["nxyz"][:, 1:]
        z = batch["nxyz"][:, 0]
        alpha, beta, gamma = torch.rand(3)
        rot = torch.Tensor([[cos(alpha) * cos(beta),
                             cos(alpha) * sin(beta) * sin(gamma)
                             - sin(alpha) * cos(gamma), cos(alpha) *
                             sin(beta) * cos(gamma)
                             + sin(alpha) * sin(gamma)],
                            [sin(alpha) * cos(beta),
                             sin(alpha) * sin(beta) * sin(gamma)
                             + cos(alpha) * cos(gamma), sin(alpha) *
                             sin(beta) * cos(gamma)
                             - cos(alpha) * sin(gamma)],
                            [-sin(beta), cos(beta) * sin(gamma),
                             cos(beta) * cos(gamma)]
                            ])

        new_xyz = torch.stack([torch.matmul(rot, i) for i in xyz])
        new_nxyz.append(torch.cat([z.reshape(-1, 1), new_xyz],
                                  dim=-1))
    new_dset.props["nxyz"] = new_nxyz

    return new_dset


def test_batch_featurize(job_dir="."):

    dset_path = ("/home/saxelrod/Repo/projects/ax_autopology"
                 "/NeuralForceField/tutorials/data/"
                 "switch_demonstration.pth.tar")

    # dset = qm_to_dset(dset_path, job_dir)
    # dset.save(dset_path)
    dset = Dataset.from_file(dset_path)

    dset = featurize_dset(dset)
    dset = generate_neighbors(dset,
                              cutoffs=CUTOFFS,
                              cutoff_names=CUTOFF_NAMES)

    # new_dset = test_rotate(dset)
    # # gives same features!
    # new_dset = featurize_dset(new_dset)
    # new_dset = generate_neighbors(new_dset,
    #                               cutoffs=CUTOFFS,
    #                               cutoff_names=CUTOFF_NAMES)

    dset.save(dset_path)

    return dset


if __name__ == "__main__":
    dset = test_batch_featurize()
