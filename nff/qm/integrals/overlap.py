"""
Functions for computing overlaps of Gaussian orbitals.
Formulas from Chapter 9 in Molecular Electronic Structure 
Theory, by Helgaker, Jorgensen and Olsen.
"""

# import numba as nb
import numpy as np
import torch
from nff.utils.scatter import compute_grad

# @nb.jit(nopython=True)


def horizontal(s, p, r_pa):

    num = s.shape[0]

    for i in range(1, num):
        s[i, :] = r_pa * s[i - 1, :].clone()

        if i > 1:
            extra = 1 / (2 * p) * ((i - 1) * s[i - 2, :])
            s[i, :] = s[i, :] + extra

    return s


# @nb.jit(nopython=True)
def vertical(s, r_pb, p):

    l_1 = s.shape[2]

    i_range = torch.arange(s.shape[1])
    r_pb_shape = r_pb.reshape(-1, 1)

    zeros = torch.zeros(3, 1)

    for j in range(1, l_1):

        s_term = r_pb_shape * s[:, :, j - 1].clone()
        new_s = torch.cat((zeros, s[:, :-1, j - 1]), dim=1)
        i_term = i_range / (2 * p) * new_s

        s[:, :, j] = (s_term + i_term)

        if j > 1:
            j_term = (j - 1) / (2 * p) * s[:, :, j - 2]
            s[:, :, j] = s[:, :, j] + j_term

    return s


# @nb.jit(nopython=True)
def get_prelims(r_a,
                r_b,
                a,
                b,):

    p = a + b
    mu = a * b / (a + b)

    r_ab = r_a - r_b
    big_p = (a * r_a + b * r_b) / p

    r_pa = big_p - r_a
    r_pb = big_p - r_b

    s_0 = np.sqrt(np.pi / p) * torch.exp(-mu * r_ab ** 2)

    return r_pa, r_pb, s_0, p


def compute_overlaps(l_0,
                     l_1,
                     p,
                     r_pa,
                     r_pb,
                     s_0):

    s = torch.zeros(l_0, 3)
    s[0, :] = s_0

    s = horizontal(s, p, r_pa)
    s_t = (s.transpose(0, 1).reshape(3, 1, -1)
           .transpose(1, 2))

    zeros = torch.zeros((3, l_0, l_1 - 1))
    s = torch.cat([s_t, zeros], dim=2)
    s = vertical(s, r_pb, p)

    return s


def pos_to_overlaps(r_a,
                    r_b,
                    a,
                    b,
                    l_0,
                    l_1):
    """
    Overlaps between the Cartesian Gaussian orbitals
    of two atoms at r_a and r_b, respectively,
    with repsective position exponents a and b,
    and maximum angular momenta l_0 and l_1.
    """

    r_pa, r_pb, s_0, p = get_prelims(r_a=r_a,
                                     r_b=r_b,
                                     a=a,
                                     b=b)

    s = compute_overlaps(l_0=l_0,
                         l_1=l_1,
                         p=p,
                         r_pa=r_pa,
                         r_pb=r_pb,
                         s_0=s_0)

    return s


def test():
    r_a = torch.Tensor([1, 2, 3])
    r_b = torch.Tensor([1.1, 1.8, 2.3])

    r_a.requires_grad = True
    r_b.requires_grad = True

    a = 1
    b = 2
    l_0 = 4
    l_1 = 5

    s = pos_to_overlaps(r_a,
                        r_b,
                        a,
                        b,
                        l_0,
                        l_1)

    grad = compute_grad(r_a, s)
    print(grad)

    loss = s.sum()
    loss.backward()

    idx_pairs = [[0, 1, 1], [0, 1, 2], [1, 2, 1]]

    # from numerical integration in Mathematica
    targs = [0.167162, 7.52983e-5, -0.0320324]
    for i, idx in enumerate(idx_pairs):
        print(idx)
        print("Predicted value: %.5e" %
              (s[idx[0], idx[1], idx[2]]))
        print("Target value: %.5e" % targs[i])


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        test()
