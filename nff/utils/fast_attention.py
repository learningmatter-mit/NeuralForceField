import torch
import numpy as np


def make_w(feat_dim,
           rand_dim):

    w = np.random.rand(rand_dim, feat_dim)
    q, r = np.linalg.qr(w)
    iid = np.random.randn(q.shape[1]).reshape(1, -1)
    orth = torch.Tensor(q / q.std() * iid)

    return orth


def phi_pos(w,
            x):

    rand_dim = w.shape[0]
    h = torch.exp(-(x ** 2).sum(-1) / 2)
    pref = h / rand_dim ** 0.5
    arg = torch.exp(torch.matmul(w, x))

    out = pref * arg

    return out
