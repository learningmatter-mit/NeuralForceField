import numba as nb
import numpy as np


# @nb.jit(nopython=True)
def numba_horizontal(s, p, r_pa):

    num = s.shape[0]

    for i in range(1, num):
        s[i, :] = r_pa * s[i - 1, :]

        if i > 1:
            extra = 1 / (2 * p) * ((i - 1) * s[i - 2, :])
            s[i, :] += extra

    return s


# @nb.jit(nopython=True)
def numba_vertical(s, r_pb, p):

    l_1 = s.shape[2]

    i_range = np.arange(s.shape[1])
    r_pb_shape = r_pb.reshape(-1, 1)

    zeros = np.zeros((3, 1))

    for j in range(1, l_1):

        s_term = r_pb_shape * s[:, :, j - 1]
        new_s = np.concatenate((zeros, s[:, :-1, j - 1]), axis=1)
        i_term = i_range / (2 * p) * new_s

        s[:, :, j] = (s_term + i_term)

        if j > 1:
            j_term = (j - 1) / (2 * p) * s[:, :, j - 2]
            s[:, :, j] += j_term

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

    s_0 = np.sqrt(np.pi / p) * np.exp(-mu * r_ab ** 2)

    return r_pa, r_pb, s_0, p


def compute_overlaps(l_0,
                     l_1,
                     p,
                     r_pa,
                     r_pb,
                     s_0):

    s = np.zeros((l_0, 3))
    s[0, :] = s_0

    s = numba_horizontal(s, p, r_pa)
    s_t = (s.transpose().reshape(3, 1, -1)
           .transpose(0, 2, 1))

    zeros = np.zeros((3, l_0, l_1 - 1))
    s = np.concatenate([s_t, zeros], axis=2)
    s = numba_vertical(s, r_pb, p)

    return s


def pos_to_overlaps(r_a,
                    r_b,
                    a,
                    b,
                    l_0,
                    l_1):

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
    r_a = np.array([1, 2, 3])
    r_b = np.array([1.1, 1.8, 2.3])
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
    idx_pairs = [[0, 1, 1], [0, 1, 2], [1, 2, 1]]

    # from numerical integration in Mathematica
    targs = [0.167162, 7.52983e-5, -0.0320324]
    for i, idx in enumerate(idx_pairs):
        print(idx)
        print("Predicted value: %.5f" %
              (s[idx[0], idx[1], idx[2]]))
        print("Target value: %.5f" % targs[i])


if __name__ == "__main__":
    test()
