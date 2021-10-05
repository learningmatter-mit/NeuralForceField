from torch.utils.data import DataLoader
import torch
import numpy as np
import copy

from nff.data import Dataset, collate_dicts
from nff.train.evaluate import evaluate


def dr_dt(p, m):
    r"""
    d\bar{R}_{j, \eta}^{J} /dt, where \bar{R}_{j, \eta}^{J}  is the average position
    of the \eta^th nuclear degree of freedom, J is the electronic state, and j =
    {1, 2, ..., N_J} enumerates the basis functions (maximum of N_J for state J).
    So there are N_j basis functions, each of which consists of \eta =
    {1, 2, ..., 3N_atoms} Gaussians.


        Args:
                p (torch.Tensor): a momentum tensor of dimension N_J x N_at x 3.
                m (torch.Tensor): a mass tensor of dimension N_at

    """

    v = p / m.reshape(1, -1, 1)
    return v


def dp_dt(en_grad):
    """
    Args:
            f (torch.Tensor): an energy gradient tensor of dimension N_J x N_at x 3.
                    It is evaluated at the center positions of each of the
                    N_j basis vectors.

    """

    f = -en_grad
    return f


def dgamma_dt(en, p, m):
    """
    Args:
            en (torch.Tensor): energy tensor of dimension N_J
            p (torch.Tensor): a momentum tensor of dimension N_J x N_at x 3
            m (torch.Tensor): a mass tensor of dimension N_at
    Returns:
            deriv (torch.Tensor): derivative of gamma, a tensor of dimension
                    N_J.
    """

    reshape_m = m.reshape(1, -1, 1)
    deriv = -en + (p ** 2 / (2 * reshape_m)).sum((1, 2))

    return deriv


def to_dset(r, atom_nums, nbrs, gen_nbrs):
    """
    Args:
            r (torch.Tensor): a position tensor of dimension N_J x N_at x 3,
            where N_j is the number of basis functions for the given state
            J and N_at is the number of atoms.
    """

    atom_num_reshape = atom_nums.reshape(-1, 1)
    nxyz = [torch.cat([atom_num_reshape, xyz], dim=-1)
            for xyz in r]

    dataset = Dataset(props={"nxyz": nxyz})

    if nbrs is None or gen_nbrs:
        dataset.generate_neighbor_list()
    else:
        dataset.props["nbr_list"] = nbrs

    return dataset


def get_engrad(r,
               atom_nums,
               nbrs,
               gen_nbrs,
               batch_size,
               device,
               model,
               diabat_keys):
    """
    Args:
            r (torch.Tensor): a position tensor of dimension N_J x N_at x 3,
            where N_j is the number of basis functions for the given state
            J and N_at is the number of atoms.
    """

    dataset = to_dset(r=r,
                      atom_nums=atom_nums,
                      nbrs=nbrs,
                      gen_nbrs=gen_nbrs)

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=collate_dicts)

    results, _, _ = evaluate(model=model,
                             loader=loader,
                             loss_fn=lambda x, y: 0,
                             device=device,
                             debatch=True)

    for key, val in results.items():
        if key.endswith("_grad") or key.startswith('nacv_'):
            results[key] = torch.stack(val)

    return results, dataset


def compute_derivs(r,
                   m,
                   atom_num,
                   p,
                   nbrs,
                   gen_nbrs,
                   batch_size,
                   device,
                   model,
                   diabat_keys,
                   diabatic):

    results, dataset = get_engrad(r,
                                  atom_num,
                                  nbrs,
                                  gen_nbrs,
                                  batch_size,
                                  device,
                                  model,
                                  diabat_keys)
    num_states = len(diabat_keys)
    derivs = []

    for i in range(num_states):

        if diabatic:
            en_key = diabat_keys[i][i]
            grad_key = en_key + "_grad"
            en = results[en_key]
            en_grad = results[grad_key]
        else:
            en = results[f"energy_{i}"]
            en_grad = results[f"energy_{i}_grad"]

        gamma_deriv = dgamma_dt(en, p, m)
        p_deriv = dp_dt(en_grad)
        r_deriv = dr_dt(p, m)

        dic = {"gamma": gamma_deriv,
               "p": p_deriv,
               "r": r_deriv}

        derivs.append(dic)

    return derivs


def overlap_formula(expand_r_i,
                    expand_r_j,
                    expand_alpha_i,
                    expand_alpha_j,
                    expand_p_i,
                    expand_p_j):

    r_i = expand_r_i.numpy()
    r_j = expand_r_j.numpy()
    alpha_i = expand_alpha_i.numpy()
    alpha_j = expand_alpha_j.numpy()
    p_i = expand_p_i.numpy()
    p_j = expand_p_j.numpy()

    A = (-alpha_j * r_j ** 2 - alpha_i * r_i ** 2
         + 1j * p_j * (-r_j) - 1j * p_i * (-r_i))

    B = alpha_i + alpha_j

    C = (2 * alpha_j * r_j + 1j * p_j
         + 2 * alpha_i * r_i - 1j * p_i)

    # has dimension N_I x N_J x N_at x 3
    overlaps = ((2 / np.pi) ** 0.5 * (alpha_i * alpha_j) ** 0.25
                * np.exp(A) * (np.pi / B) ** 0.5 * np.exp(C ** 2 / (4 * B)))

    # take the product over the last two dimensions
    N_I = expand_r_i.shape[0]
    N_j = expand_r_i.shape[1]

    # ** is this one actually a product or a sum?!?!?! I think it's actually a product
    # right?

    # N_I x N_J
    overlap_prod = overlaps.reshape(N_I, N_j, -1).prod(-1)

    return overlap_prod


def tile_params(r_i,
                r_j,
                p_i,
                p_j,
                alpha_i,
                alpha_j,
                m_i=None,
                m_j=None):

    N_I = r_i.shape[0]
    N_J = r_i.shape[0]
    N_at = r_i.shape[1]

    expand_r_i = r_i.expand(N_J, N_I, N_at, 3).transpose(0, 1)
    expand_r_j = r_j.expand(N_I, N_J, N_at, 3)

    expand_p_i = p_i.expand(N_J, N_I, N_at, 3).transpose(0, 1)
    expand_p_j = p_j.expand(N_I, N_J, N_at, 3)

    expand_alpha_i = (alpha_i.reshape(1, 1, N_at, 1)
                      .expand(N_I, N_J, N_at, 3))

    expand_alpha_j = (alpha_j.reshape(1, 1, N_at, 1)
                      .expand(N_J, N_I, N_at, 3)
                      .transpose(0, 1))

    if m_i is not None and m_j is not None:
        expand_mi = (m_i.reshape(1, 1, N_at, 1)
                     .expand(N_I, N_J, N_at, 3))

        expand_mj = (m_j.reshape(1, 1, N_at, 1)
                     .expand(N_J, N_I, N_at, 3)
                     .transpose(0, 1))

        return (expand_r_i, expand_r_j, expand_p_i,
                expand_p_j, expand_alpha_i, expand_alpha_j,
                expand_mi, expand_mj)
    else:
        return (expand_r_i, expand_r_j, expand_p_i,
                expand_p_j, expand_alpha_i, expand_alpha_j)


def get_overlaps(r_i,
                 r_j,
                 alpha_i,
                 alpha_j,
                 p_i,
                 p_j):
    """
    Args:
            r_i: Gaussian positions in state i. Tensor of
                    shape N_I x N_at x 3.
            alpha_i: Alpha's of state i, dimension N_at
            r_j: Gaussian positions in state j. Tensor of
                    shape N_J x N_at x 3.
            alpha_j: Alpha's of state j, dimension N_at

    """

    (expand_r_i, expand_r_j, expand_p_i,
     expand_p_j, expand_alpha_i, expand_alpha_j) = tile_params(r_i,
                                                               r_j,
                                                               p_i,
                                                               p_j,
                                                               alpha_i,
                                                               alpha_j)

    r_max = ((expand_alpha_i * expand_r_i + expand_alpha_j * expand_r_j)
             / (expand_alpha_i + expand_alpha_j))

    # G_ij

    overlap = overlap_formula(expand_r_i=expand_r_i,
                              expand_r_j=expand_r_j,
                              expand_alpha_i=expand_alpha_i,
                              expand_alpha_j=expand_alpha_j,
                              expand_p_i=expand_p_i,
                              expand_p_j=expand_p_j)

    return overlap, r_max


def get_coupling_r(r_list,
                   p_list,
                   alpha_dic,
                   atom_nums,
                   min_overlap):
    """
    Get all overlaps betwene nuclear wave functions on different states, and get the positions
    at which the overlaps are large enough that we'll want to calculate matrix elements
    between them (i.e. non-adiabatic or diabatic couplings).

    Args:
            "*_list": list of the quantities on each electronic state
            alpha_dic (dict): dictionary that gives you the fixed Gaussians
                wave packet width for a given atomic number
    """

    num_states = len(r_list)
    couple_dic = {}

    for i in range(num_states):
        for j in range(num_states):

            r_i = r_list[i]  # N_I x N_at x 3
            r_j = r_list[j]  # N_J x N_at x 3

            p_i = p_list[i]
            p_j = p_list[j]

            alpha_i = torch.Tensor([alpha_dic[atom_num]
                                    for atom_num in atom_nums])
            alpha_j = copy.deepcopy(alpha_i)

            overlap, r_max = get_overlaps(r_i=r_i,
                                          r_j=r_j,
                                          alpha_i=alpha_i,
                                          alpha_j=alpha_j,
                                          p_i=p_i,
                                          p_j=p_j)

            couple_mask = abs(overlap) > min_overlap
            couple_idx = couple_mask.nonzero()
            couple_r = r_max[couple_idx[:, 0], couple_idx[:, 1]]

            couple_dic[f"{i}_{j}"] = {"overlap": overlap,
                                      "couple_idx": couple_idx,
                                      "couple_r": couple_r,
                                      "couple_mask": couple_mask}

    return couple_dic


def compute_A(m, nacv, hbar=1):
    """
    Compute the non-adiabatic coupling term A that shows up in the off-diagonal
    of the Hamiltonian in the adiabatic basis.

    Args:
            m (torch.Tensor): masses of dimension N_at,
            nacv (torch.Tensor): non-adiabatic coupling vector, of dimension
                    n_ij x N_at x 3. n_ij are the subsets of of N_I and N_J with enough
                    overlap to warrant a matrix element calculation.
            hbar (float): reduced Planck constant. Set to 1 when using atomic units.
    Returns:
            A_ij (torch.Tensor): coupling term A, of dimension n_ij.
    """

    m_reshape = m.reshape(1, -1, 1)
    A_ij = (-hbar ** 2 / m_reshape * nacv).sum((1, 2))

    return A_ij


def nonad_ham_ij(couple_r,
                 nacv,
                 overlap,
                 mask,
                 alpha_j,
                 r_j,
                 p_j,
                 m):
    """
    Construct the off-diagonal elements of the Hamiltonian in the
    adiabatic basis. Computes f_ij = f(R_ij), the value of the matrix
    element at the centroid positions R_ij, and multiplies it with the
    nuclear wave function overlaps.

    Args:
        couple_r (torch.Tensor): Tensor of centroid positions between
                the basis functions. It has dimension (n_ij) x N_at x 3;
                n_ij are the subsets of of N_I and N_J with enough overlap
                to warrant a matrix element calculation.

        nacv (torch.Tensor): n_ij x N_at x 3 non-adiabatic coupling
                        vector.

        overlap (np.array): N_I x N_J overlap matrix between
                        Gaussian basis functions.

        mask (np.array): N_I x N_J mask that is True for n_ij elements.

        alpha_j (torch.Tensor): N_at Gaussian widths for state J
        r_j (torch.Tensor): N_J x N_at x 3 positions for state J
        p_j (torch.Tensor): N_J x N_at x 3 momenta for state J

    Returns:

        h_ij (np.array): N_I x N_J Hamiltonian matrix element along
                electronic dimensions i and j, when i != j.

    """

    # non-zero n_ij indices of the mask
    idx = torch.LongTensor(mask).nonzero()

    # Take the values of r_j along the mask. This creates an
    # n_ij x N_at x 3 dimensional tensor
    mask_rj = r_j[idx[:, 1]]

    # Do the same for p_j
    mask_pj = p_j[idx[:, 1]]

    # reshape alpha_j to align with the n_ij x N_at x 3 dimensions
    alpha_j_reshape = alpha_j.reshape(1, -1, 1)

    # Compute the nabla part of the matrix element
    # Since the atomic basis functions are multiplied together,
    # we take the sum along the atomic dimensions

    n_ij = idx.shape[0]
    nabla_ij_real = ((-alpha_j_reshape * (couple_r - mask_rj))
                     .reshape(n_ij, -1).sum(-1))
    nabla_ij_im = mask_pj.reshape(n_ij, -1).sum(-1)

    # Convert to numpy to make it complex

    nabla_ij = nabla_ij_real.numpy() + 1j * nabla_ij_im.numpy()

    # Multiply with A_ij to get f_ij (dimension n_ij)

    A_ij = compute_A(m, nacv).numpy()
    f_ij = A_ij * nabla_ij

    # compute final matrix elements h_ij, of dimension
    #  N_I x N_J

    h_ij = np.zeros_like(overlap)
    h_ij[mask] = f_ij * overlap[mask]

    return h_ij


# def nuc_ke(r_j,
#            p_j,
#            alpha_j,
#            couple_r,
#            overlap,
#            mask,
#            m,
#            hbar=1):
#     """
#     Get the diagonal kinetic energy part of the Hamiltonian.
#     Args:

#         couple_r (torch.Tensor): Tensor of centroid positions between
#                 the basis functions. It has dimension (n_ij) x N_at x 3;
#                 n_ij are the subsets of of N_I and N_J with enough overlap
#                 to warrant a matrix element calculation.
#         overlap (np.array): N_I x N_J overlap matrix between
#                         Gaussian basis functions.
#         mask (np.array): N_I x N_J mask that is True for n_ij elements.
#         m (torch.Tensor): masses of dimension N_at,

#     """


#     #  **** this should actually be done analytically


#     # non-zero n_ij indices of the mask
#     idx = torch.LongTensor(mask).nonzero()

#     # Take the values of r_j along the mask. This creates an
#     # n_ij x N_at x 3 dimensional tensor
#     mask_rj = r_j[idx[:, 1]].numpy()

#     # Do the same for p_j
#     mask_pj = p_j[idx[:, 1]].numpy()

#     # reshape alpha_j to align with the n_ij x N_at x 3 dimensions
#     alpha_j_reshape = alpha_j.reshape(1, -1, 1).numpy()

#     # same for m
#     m_reshape = m.reshape(1, -1, 1).numpy()

#     # convert couple_r to numpy

#     r_ij = couple_r.numpy()

#     # Vector form of f_ij
#     vec_f_ij = (- hbar ** 2 / (2 * m_reshape) *
#                 (-2 * alpha_j_reshape +
#                  (1j * mask_pj + 2 * alpha_j_reshape
#                   * (mask_rj - r_ij)) ** 2))

#     # Take the product along the coordinate dimensions to get f_ij,
#     # of dimension n_ij

#     n_ij = idx.shape[0]
#     f_ij = vec_f_ij.reshape(n_ij, -1).prod(-1)

#     h_ij = np.zeros_like(overlap)
#     h_ij[mask] = f_ij * overlap[mask]

#     return h_ij

def nuc_ke(r_j,
           p_j,
           alpha_j,
           r_i,
           p_i,
           alpha_i,
           mask,
           m,
           hbar=1):
    """
    Get the diagonal kinetic energy part of the Hamiltonian.
    Args:

        couple_r (torch.Tensor): Tensor of centroid positions between
                the basis functions. It has dimension (n_ij) x N_at x 3;
                n_ij are the subsets of of N_I and N_J with enough overlap
                to warrant a matrix element calculation.
        overlap (np.array): N_I x N_J overlap matrix between
                        Gaussian basis functions.
        mask (np.array): N_I x N_J mask that is True for n_ij elements.
        m (torch.Tensor): masses of dimension N_at,

    """

    #  **** this should actually be done analytically

    (expand_r_i, expand_r_j, expand_p_i,
     expand_p_j, expand_alpha_i, expand_alpha_j,
     expand_mi, expand_mj) = tile_params(r_i,
                                         r_j,
                                         p_i,
                                         p_j,
                                         alpha_i,
                                         alpha_j)

    A = (-2 * expand_alpha_j - (expand_p_j) ** 2
         + 4 * 1j * expand_alpha_j * expand_p_j * expand_r_j)

    B = (-4 * 1j * expand_alpha_j * expand_p_j
         - 8 * (expand_alpha_j) ** 2 * expand_r_j)

    C = 4 * expand_alpha_j ** 2

    D = (1j * expand_p_i * expand_r_i - 1j * expand_p_j * expand_r_j
         - expand_alpha_i * expand_r_i ** 2 - expand_alpha_j * expand_r_j ** 2)

    E = (1j * expand_p_j - 1j * expand_p_i + 2 * expand_r_i * expand_alpha_i
         + 2 * expand_r_j * expand_alpha_j)

    F = expand_alpha_i + expand_alpha_j
    #  *** are we dividing by the right mass here??

    prefactor = ((2) ** 0.5 * (expand_alpha_i * expand_alpha_j) ** 0.25
                 * (-hbar ** 2) / (2 * expand_mj))

    main_term = (1 / (4 * F ** (5/2))
                 * np.exp(D + E ** 2 / (4 * F))
                 * (C * E ** 2 + 2 * (C + B * E) * F + 4 * A * F ** 2))

    # dimension N_I x N_J x N_at x 3
    ke_vec = prefactor * main_term

    # actual kinetic energy is the sum over last two dimensions

    ke = ke_vec.reshape(ke_vec.shape[0], ke_vec.shape[1], -1).sum(-1)

    return ke


def elec_e(energies,
           overlap,
           mask):
    """
        Args:
        energies (torch.Tensor): n_ij dimensional tensor
                of energies for the given state, calculated
                at the set of n_ij relevant centroids.
    overlap (np.array): N_I x N_J overlap matrix between
                    Gaussian basis functions.
    mask (np.array): N_I x N_J mask that is True for n_ij elements.

    """

    h_ij = np.zeros_like(overlap)
    h_ij[mask] = energies.reshape(-1) * overlap[mask]

    return h_ij


def construct_ham(r_list,
                  p_list,
                  atom_nums,
                  m,
                  couple_dic,
                  nbrs,
                  gen_nbrs,
                  batch_size,
                  device,
                  model,
                  diabat_keys,
                  alpha_dic):
    """
    This needs to be fixed  -- need properly H term
    for adiabatic, and also nuclear kinetic energy
    for nuclei for both adiabatic and diabatic

    Args:
        r_list (list[torch.Tensor]): list of wavepacket
                positions for each state
        p_list (list[torch.Tensor]): list of wavepacket
                momenta for each state.
        atom_nums (torch.Tensor): atomic numbers of dimension N_at,
        m (torch.Tensor): masses of dimension N_at,

    Returns:
        h_d (np.array): Hamiltonian in diabatic basis (complex)
        h_ad (np.array): Hamiltonian in adiabatic basis
    """

    num_states = int(couple_dic ** 0.5)
    max_basis = max([r.shape[0] for r in r_list])

    # padded, as different states have different number of
    # trj basis functions

    h_d = torch.zeros(num_states, num_states,
                      max_basis, max_basis).numpy()

    h_ad = torch.zeros(num_states, num_states,
                       max_basis,  max_basis).numpy()

    for key, sub_dic in couple_dic.items():

        i, j = key.split("_")
        couple_r = sub_dic["couple_r"]

        results, dataset = get_engrad(r=couple_r,
                                      atom_nums=atom_nums,
                                      nbrs=nbrs,
                                      gen_nbrs=gen_nbrs,
                                      batch_size=batch_size,
                                      device=device,
                                      model=model,
                                      diabat_keys=diabat_keys)

        mask = sub_dic["couple_mask"]  # numpy array
        overlap = sub_dic["overlap"]  # numpy array (complex)

        # h_d_ij = torch.zeros_like(overlap).numpy()

        alpha_j = torch.Tensor([alpha_dic[atom_num]
                                for atom_num in atom_nums])
        alpha_i = copy.deepcopy(alpha_j)

        r_j = r_list[j]
        p_j = p_list[j]

        r_i = r_list[i]
        p_i = p_list[i]

        if i == j:

            # nuclear kinetic energy component

            h_ad_ij = nuc_ke(r_j,
                             p_j,
                             alpha_j,
                             r_i,
                             p_i,
                             alpha_i,
                             mask,
                             m)

            h_d_ij = copy.deepcopy(h_ad_ij)

            # Electronic Hamiltonian component

            ad_key = f"energy_{i}"
            diabat_key = diabat_keys[i][j]

            h_ad_ij += elec_e(results[ad_key],
                              overlap,
                              mask)

            h_d_ij += elec_e(results[diabat_key],
                             overlap,
                             mask)

        else:

            # The off-diagonal Hamiltonian in the adiabatic
            # basis involves the non-adiabatic coupling vector

            nacv = results[f"nacv_{i}{j}"]
            h_ad_ij = nonad_ham_ij(couple_r=couple_r,
                                   nacv=nacv,
                                   overlap=overlap,
                                   mask=mask,
                                   alpha_j=alpha_j,
                                   r_j=r_j,
                                   p_j=p_j,
                                   m=m)

            # The off-diagonal Hamiltonian in the diabatic
            # basis is the diabatic eletronic energy

            diabat_key = diabat_keys[i][j]
            h_d_ij = elec_e(results[diabat_key],
                            overlap,
                            mask)

        N_I, N_J = overlap.shape[:2]
        h_d[i, j, :N_I, :N_J] = h_d_ij
        h_ad[i, j, :N_I, :N_J] = h_ad_ij

    return h_d, h_ad


def diabat_spawn_criterion(states,
                           results,
                           diabat_keys,
                           threshold):
    """
        Args:

          results_list (list[dict]): list of dictionaries. Each
                            dictionary corresponds to an electronic state.
                            It contains model predictions for the
                            positions of the nuclear wave packets on
                             that state.
        Returns:
                thresh_dic (dict): dictionary with keys for each state,
                        the value of which is a subdictionary. The subdictionary
                        contains keys for each other state. Say we're looking at 
                        main key i and subdictionary key j. Then thresh_dic[i][j]
                        is a boolean tensor of dimension N_I. For each Gaussian
                        basis function in state i, it tells you whether you should
                        replicate it on state j.

    """

    thresh_dic = {i: {} for i in range(len(states))}

    for i in states:
        for j in states:
            if i == j:
                continue
            diabat_key = diabat_keys[i][j]

            # Diabatic coupling from state i to state j,
            # for the positions of the nuclear wave packets
            # currently on state i.

            # Dimension = N_I
            diabat_coup = results[diabat_key]

            # Difference between diagonal diabatic elements

            diabat_key_i = diabat_keys[i][i]
            diabat_key_j = diabat_keys[j][j]
            delta_diabat = results[diabat_key_i] - results[diabat_key_j]

            # Effective coupling

            h_eff = diabat_coup / delta_diabat
            thresh_dic[i][j] = abs(h_eff) > threshold

    return thresh_dic


def adiabat_spawn_criterion(states,
                            results,
                            v_list,
                            threshold):
    """
        Args:

          results_list (list[dict]): list of dictionaries. Each
                            dictionary corresponds to an electronic state.
                            It contains model predictions for the
                            positions of the nuclear wave packets on
                             that state.
            v_list (list[torch.Tensor]): list of velocities for wave
                                        packets on each state.
        Returns:
                thresh_dic (dict): dictionary with keys for each state,
                        the value of which is a subdictionary. The subdictionary
                        contains keys for each other state. Say we're looking at 
                        main key i and subdictionary key j. Then thresh_dic[i][j]
                        is a boolean tensor of dimension N_I. For each Gaussian
                        basis function in state i, it tells you whether you should
                        replicate it on state j.

    """

    thresh_dic = {i: {} for i in range(len(states))}

    for i in states:
        for j in states:
            if i == j:
                continue

            # Velocity, dimension N_I x N_at x 3
            vel = v_list[i]

            # Non-adiabatic coupling from state i to state j,
            # for the positions of the nuclear wave packets
            # currently on state i.

            # Dimension = N_I x N_at x 3
            nacv = results[f"nacv_{i}{j}"]

            # Effective coupling

            h_eff = (vel * nacv).sum()
            thresh_dic[i][j] = {"criterion": (abs(h_eff) > threshold).any(),
                                "val": h_eff.norm()}

    return thresh_dic


def get_vals(diabatic,
             surf,
             diabat_keys,
             results):

    i = surf
    if diabatic:
        en_key = diabat_keys[i][i]
        grad_key = en_key + "_grad"
        en = results[en_key]
        en_grad = results[grad_key]
    else:
        en = results[f"energy_{i}"]
        en_grad = results[f"energy_{i}_grad"]

    return en, en_grad


def nuc_classical(r,
                  gamma,
                  m,
                  atom_num,
                  p,
                  nbrs,
                  gen_nbrs,
                  batch_size,
                  device,
                  model,
                  states,
                  diabat_keys,
                  diabatic,
                  dt,
                  surf,
                  old_results):

    # classical propagation of nuclei

    old_en, old_grad = get_vals(diabatic=diabatic,
                                surf=surf,
                                diabat_keys=diabat_keys,
                                results=old_results)

    # note: we need a p + 1/2 dt and a p + 3/2 dt
    # The p that we keep track of will always be 1/2 dt
    # ahead of r. Need to initialize it that way on step 1.
    # (????? That also means that this gamma thing should be
    # using p_t and p_(t+dt), but it's really using p_(t+dt/2)
    # and p_t(t + 3 dt/2))

    dgamma = 1 / 2 * dgamma_dt(old_en, p, m) * dt

    # r and p have dim N_J x N_at x 3
    # m has dim N_at
    r_new = r + 1 / m.reshape(1, -1, 1) * dt * p

    results, dataset = get_engrad(r_new,
                                  atom_num,
                                  nbrs,
                                  gen_nbrs,
                                  batch_size,
                                  device,
                                  model,
                                  diabat_keys)

    new_en, new_grad = get_vals(diabatic=diabatic,
                                surf=surf,
                                diabat_keys=diabat_keys,
                                results=results)

    p_new = p - new_grad * dt

    dgamma += 1 / 2 * dgamma_dt(new_en, p_new, m) * dt
    gamma_new = gamma + dgamma

    return r_new, p_new, gamma_new, results


def find_spawn(r,
               gamma,
               m,
               atom_num,
               p,
               nbrs,
               gen_nbrs,
               batch_size,
               device,
               model,
               diabat_keys,
               diabatic,
               dt,
               new_surf,
               old_results,
               old_surf,
               threshold):

    # classical propagation

    too_big = False
    couplings = []
    r_new = copy.deepcopy(r)
    p_new = copy.deepcopy(p)
    gamma_new = copy.deepcopy(gamma)

    r_list = []
    p_list = []
    gamma_list = []
    old_results_list = []

    while too_big:
        # this is right: keep propagating along the old surface
        r_new, p_new, gamma_new, new_results = nuc_classical(r_new,
                                                             gamma_new,
                                                             m,
                                                             atom_num,
                                                             p_new,
                                                             nbrs,
                                                             gen_nbrs,
                                                             batch_size,
                                                             device,
                                                             model,
                                                             diabat_keys,
                                                             diabatic,
                                                             dt,
                                                             old_surf,
                                                             old_results)

        states = [old_surf, new_surf]

        if diabatic:
            spawn_dic = diabat_spawn_criterion(states,
                                               new_results,
                                               diabat_keys,
                                               threshold)
        else:
            spawn_dic = adiabat_spawn_criterion(states,
                                                new_results,
                                                v_list,
                                                threshold)

        coupling = spawn_dic[old_surf][new_surf]['val']
        couplings.append(coupling)
        too_big = spawn_dic[old_surf][new_surf]['criterion']

        r_list.append(r_new)
        p_list.append(p_new)
        gamma_list.append(gamma_new)

        old_results = new_results
        old_results_list.append(copy.deepcopy(old_results))

    couplings = np.array(couplings)
    spawn_idx = couplings.argmax()

    spawn_r = r_list[spawn_idx]
    # this will need to be readjusted
    spawn_p = p_list[spawn_idx]
    # spawn_gamma = gamma_list[spawn_idx]

    return spawn_r, spawn_p, spawn_idx, old_results_list


def rescale(p_new,
            m,
            diabatic,
            results,
            old_surf,
            new_surf):

    if diabatic:
        raise NotImplementedError
    else:
        # p has dimension N_J x N_at x 3
        # nacv has dimension N_J x N_at x 3

        nacv = results[f'nacv_{old_surf}{new_surf}']
        norm = (nacv ** 2).sum(-1) ** 0.5
        nacv_unit = nacv / norm

        # dot product
        projection = (nacv_unit * p_new).sum(-1)

        # p_parallel
        N_J, N_at = projection.shape
        p_par = (projection.reshape(N_J, N_at, 1)
                 * nacv_unit)

        # p perpendicular
        p_perp = p_new - p_par

        # get energies before and after hop
        # m has shape N_at
        # is this right?

        t_old = (p_new ** 2 / (2 * m.reshape(1, -1, 1))).sum()
        t_old_perp = (p_perp ** 2 / (2 * m.reshape(1, -1, 1))).sum()
        t_old_par = (p_par ** 2 / (2 * m.reshape(1, -1, 1))).sum()
        v_old = results[f'energy_{old_surf}']
        v_new = results[f'energy_{new_surf}']

        # re-scale p_parallel
        # not 100% sure if this is right

        scale_sq = (t_old + v_old - (t_old_perp + v_new)) / t_old_par

        if scale_sq < 0:
            # kinetic energy can't compensate the change in
            # potential energy
            return None

        scale = scale_sq ** 0.5

        new_p = p_par * scale + p_perp

    return new_p


def backward_prop(spawn_r,
                  spawn_p,
                  spawn_gamma,
                  spawn_idx,
                  m,
                  atom_num,
                  nbrs,
                  gen_nbrs,
                  batch_size,
                  device,
                  model,
                  diabat_keys,
                  diabatic,
                  dt,
                  new_surf,
                  old_results_list,
                  old_surf,
                  threshold,
                  dr):

    num_steps = spawn_idx
    old_results = old_results_list[spawn_idx]

    r_new = copy.deepcopy(spawn_r)
    p_new = copy.deepcopy(spawn_p)
    # re-scale the momentum along the nacv
    # to ensure energy conservation
    p_new = rescale(p_new=p_new,
                    m=m,
                    diabatic=diabatic,
                    results=old_results,
                    old_surf=old_surf,
                    new_surf=new_surf)

    if p_new is None:

        grad = old_results[f'energy_{new_surf}_grad']
        r_new = spawn_r - grad * dr

        new_results = get_engrad(r_new=r_new,
                                 atom_num=atom_num,
                                 nbrs=nbrs,
                                 gen_nbrs=gen_nbrs,
                                 batch_size=batch_size,
                                 device=device,
                                 model=model,
                                 diabat_keys=diabat_keys)

        old_results_list[spawn_idx] = new_results

        return backward_prop(r_new,
                             spawn_p,
                             spawn_gamma,
                             spawn_idx,
                             m,
                             atom_num,
                             nbrs,
                             gen_nbrs,
                             batch_size,
                             device,
                             model,
                             diabat_keys,
                             diabatic,
                             dt,
                             new_surf,
                             old_results_list,
                             old_surf,
                             threshold,
                             dr)

    gamma_new = copy.deepcopy(spawn_gamma)

    # We may want to implement N_s >= 1, i.e. potentially more than
    # one basis function spawned in the crossing region. This could
    # be important for the diabatic representation.

    for _ in range(num_steps):
        # I don't think just replacing it with -dt is right - don't
        # we have to replace the forces with their negative values?
        r_new, p_new, gamma_new, new_results = nuc_classical(r_new,
                                                             gamma_new,
                                                             m,
                                                             atom_num,
                                                             p_new,
                                                             nbrs,
                                                             gen_nbrs,
                                                             batch_size,
                                                             device,
                                                             model,
                                                             diabat_keys,
                                                             diabatic,
                                                             (-dt),
                                                             new_surf,
                                                             old_results)
        old_results = new_results

    return r_new, p_new


# Next up:
# 1. Add spawning rejection criterion
# 2. Replace approximate diagonal KE with exact
# 3. Are we letting all the exp(i gamma) prefactors
# run around untracked?
