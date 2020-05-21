import copy
import torch
from nff.io.ase import AtomsBatch
from ase.md.velocitydistribution import Stationary, ZeroRotation
# import soaplite
# from soaplite import genBasis
from sklearn import preprocessing
import pdb

from nff.nn.layers import gaussian_smearing


def make_smears(nxyz, n_gaussians, channels, start, stop):
    """
    Returns:
        channel_xyz_smears: [x_smears, y_smears, z_smears],
            where x_smears = [x_smear_channel_0, x_smear_channel_1, 
            ..., x_smear_channel_N]
    """

    xyz = torch.cat([nxyz[:, 1], nxyz[:, 2], nxyz[:, 3]]).reshape(-1, 1)

    offset = torch.linspace(start, stop, n_gaussians)
    widths = torch.FloatTensor(
        (offset[1] - offset[0]) * torch.ones_like(offset))

    smears = gaussian_smearing(xyz, offset, widths, centered=False)
    xyz_smears = torch.split(smears, [len(nxyz)] * 3)
    channel_xyz_smears = []

    for i, smear in enumerate(xyz_smears):
        channel_xyz_smears.append([])
        for channel in channels:

            channel_mask = nxyz[:, 0] == channel
            zero_idx = torch.logical_not(channel_mask).nonzero()

            new_smear = copy.deepcopy(smear)
            if channel != 0:
                new_smear[zero_idx] *= 0

            channel_xyz_smears[-1].append(new_smear)

    return channel_xyz_smears


def fp_from_smears(channel_xyz_smears):

    voxel_list = []
    num_channels = len(channel_xyz_smears[0])

    num_atoms = channel_xyz_smears[0][0].shape[0]
    num_gauss = channel_xyz_smears[0][0].shape[1]

    for j in range(num_channels):

        x_smears = channel_xyz_smears[0][j]
        y_smears = channel_xyz_smears[1][j]
        z_smears = channel_xyz_smears[2][j]

        dim_x = x_smears.shape[-1]
        dim_y = y_smears.shape[-1]
        dim_z = z_smears.shape[-1]

        voxels = torch.zeros((dim_x, dim_y, dim_z))
        # voxels = torch.zeros((dim_x, dim_y))

        for i in range(num_atoms):

            x_smear = x_smears[i]
            y_smear = y_smears[i]
            z_smear = z_smears[i]

            xy_mat = torch.ger(x_smear, y_smear)
            xyz_mat = torch.ger(xy_mat.reshape(-1).double(), z_smear.double()
                                ).reshape(*[num_gauss] * 3)

            voxels += xyz_mat

            # voxels += xy_mat

        voxel_list.append(voxels)

    fp = torch.cat(voxel_list, dim=1).reshape(-1)

    return fp


def make_vox_fps(nxyz_list, n_gaussians, start, stop, use_channels=True):

    print("Starting")

    # channels don't work properly because we average the z component of
    # the nxyz

    if use_channels:
        channels = list(set(torch.cat(nxyz_list)[:, 0].tolist()))
        channels.append(0)
    else:
        channels = [0]

    fps = []
    # for i, nxyz in tqdm(enumerate(nxyz_list)):
    for i, nxyz in enumerate(nxyz_list):

        print(i)

        channel_xyz_smears = make_smears(nxyz=nxyz,
                                         n_gaussians=n_gaussians,
                                         channels=channels,
                                         start=start,
                                         stop=stop)
        fp = fp_from_smears(channel_xyz_smears)
        fps.append(fp)
        # sleep(0.1)

    fps = torch.stack(fps)

    return fps


def split_fps(fp, nxyz_list):

    channels = list(set(torch.cat(nxyz_list)[:, 0].tolist()))
    channels.append(0)
    num_channels = len(channels)
    fp_dim = fp.shape[-1] // num_channels

    splits = torch.split(fp, [fp_dim] * num_channels)

    return splits, channels


def check_box(nxyz_list, start, stop):

    all_nxyz = torch.cat(nxyz_list)
    min_r = all_nxyz[:, 1:].min()
    max_r = all_nxyz[:, 1:].max()

    if start is None or stop is None:

        print(("Using a minimum r of %.2f and max r of %.2f Angstrom "
               ) % (min_r, max_r))

        return min_r, max_r

    print(("Requested minimum r %.2f and max r %.2f; "
           "actual min and max are %.2f and %.2f"
           ) % (start, stop, min_r, max_r))

    if min_r < start or max_r > start:
        start = min_r
        stop = max_r
        print(("Changing min and max to %.2f and %.2f") %
              (min_r, max_r))

    return start, stop


def reposition_xyz(dataset, sum_weight=False):

    nxyz_list = copy.deepcopy(dataset.props["nxyz"])
    num_atoms_list = dataset.props["num_atoms"]
    mol_sizes = dataset.props.get("mol_size",
                                  num_atoms_list)
    weight_list = dataset.props.get("weights",
                                    torch.ones_like(num_atoms_list))

    for i in range(len(nxyz_list)):

        mol_size = mol_sizes[i].item()
        num_atoms = num_atoms_list[i].item()
        num_confs = num_atoms // mol_size
        N = [mol_size] * num_confs

        conf_nxyz_list = torch.split(nxyz_list[i], N)
        weights = weight_list[i]

        weighted_conf_nxyz = []

        for conf_nxyz, weight in zip(conf_nxyz_list,
                                     weights):

            numbers = conf_nxyz[:, 0]
            positions = conf_nxyz[:, 1:]
            atoms = AtomsBatch(numbers=numbers,
                               positions=positions)
            Stationary(atoms)
            ZeroRotation(atoms)

            nxyz = torch.tensor(atoms.get_nxyz())

            if sum_weight:
                weighted_conf_nxyz.append(weight * nxyz)
            else:
                weighted_conf_nxyz.append(nxyz)

        new_nxyz = torch.stack(weighted_conf_nxyz)

        if sum_weight:
            new_nxyz = new_nxyz.sum(0)
            delta = (new_nxyz - conf_nxyz[0]).abs().mean()

            print(("MAD of %.2f Angstrom between weighted sum and "
                   "lowest E conformer" % delta))

        nxyz_list[i] = new_nxyz

    return nxyz_list


def add_voxels(dataset, n_gaussians, start=None, stop=None, use_channels=True,
               use_weights=True, normalized=False):

    nxyz_list = reposition_xyz(dataset=dataset, sum_weight=True)

    start, stop = check_box(nxyz_list=nxyz_list,
                            start=start,
                            stop=stop)

    fps = make_vox_fps(nxyz_list=nxyz_list,
                       n_gaussians=n_gaussians,
                       start=start,
                       stop=stop,
                       use_channels=use_channels)

    dataset.props["voxel"] = fps


def gen_soap_params(nxyz_list, resolution):

    max_dists = []
    atom_types = []

    for conf_nxyz_list in nxyz_list:
        for nxyz in conf_nxyz_list:
            xyz = nxyz[:, 1:]
            n = xyz.size(0)
            delta = (xyz.expand(n, n, 3) - xyz.expand(n, n, 3).transpose(0, 1))
            max_dist = delta.pow(2).sum(dim=2).sqrt().max().item()
            max_dists.append(max_dist)

            atom_types += list(set(nxyz[:, 0].tolist()))

    rCut = torch.tensor(max_dists).max()
    NradBas = int(rCut / resolution)
    atom_types = list(set(atom_types))

    print(("Using a cutoff of %.2f Angstrom, %d basis functions, "
           "and %d different atom types." % (rCut, NradBas, len(atom_types))))

    # return rCut, NradBas, atom_types

    if rCut > 12:
        scale = 12 / rCut
        rCut *= scale

        print("Scaling down by %.2f to %.2f" % (scale, rCut))
    else:
        scale = 1

    return rCut, NradBas, atom_types, scale


    # return 10, 10, atom_types

def add_mol_soap(dataset, Lmax, resolution=1.0, channels=True):

    nxyz_list = reposition_xyz(dataset=dataset, sum_weight=False)
    num_atoms_list = dataset.props["num_atoms"]
    weight_list = dataset.props.get("weights",
                                    torch.ones_like(num_atoms_list))

    rCut, NradBas, atom_types, scale = gen_soap_params(nxyz_list=nxyz_list,
                                                resolution=resolution)

    myAlphas, myBetas = genBasis.getBasisFunc(rCut, NradBas)

    fps = []

    for i, conf_nxyz_list in enumerate(nxyz_list):

        conf_fps = []
        weights = weight_list[i]

        for j, nxyz in enumerate(conf_nxyz_list):

            numbers = nxyz[:, 0]
            positions = nxyz[:, 1:] * scale
            atoms = AtomsBatch(numbers=numbers,
                               positions=positions)
            com = [0, 0, 0]

            fp = torch.tensor(soaplite.get_soap_locals(
                obj=atoms,
                Hpos=com,
                alp=myAlphas,
                bet=myBetas,
                rCut=rCut,
                Lmax=Lmax,
                all_atomtypes=atom_types)[0])

            weight = weights[j].item()
            conf_fps.append(fp * weight)

        fp = torch.stack(conf_fps).sum(0)
        fps.append(fp)

    if not channels:
        for i, fp in enumerate(fps):
            feat_dim = len(fp) // len(atom_types)
            splits = fp.reshape(-1, feat_dim)
            new_fp = splits.sum(0)
            fps[i] = new_fp

    fps = torch.stack(fps)
    fps = torch.tensor(preprocessing.scale(fps))
    dataset.props["mol_soap"] = fps
