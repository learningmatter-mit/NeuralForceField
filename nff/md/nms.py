from tqdm import tqdm
import numpy as np
import os
import copy
import pickle
from rdkit import Chem
import shutil
from torch.utils.data import DataLoader
from torch.nn.modules.container import ModuleDict

from ase import optimize, units
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory as AseTrajectory
from ase.vibrations import Vibrations
from ase.units import kg, kB, mol, J, m
from ase.thermochemistry import IdealGasThermo

from nff.io.ase_ax import NeuralFF, AtomsBatch
from nff.train import load_model
from nff.data import collate_dicts, Dataset
from nff.md import nve
from nff.utils.constants import FS_TO_AU, ASE_TO_FS, EV_TO_AU, BOHR_RADIUS
from nff.utils import constants as const
from nff.nn.tensorgrad import get_schnet_hessians
from nff.utils.cuda import batch_to
from nff.nn.models.schnet import SchNet
from nff.nn.tensorgrad import hess_from_atoms as analytical_hess

PT = Chem.GetPeriodicTable()
PERIODICTABLE = PT

HA2J = 4.359744E-18
BOHRS2ANG = 0.529177
SPEEDOFLIGHT = 2.99792458E8
AMU2KG = 1.660538782E-27

TEMP = 298.15
PRESSURE = 101325
IMAG_CUTOFF = -100  # cm^-1
ROTOR_CUTOFF = 50  # cm^-1
CM_TO_EV = 1.2398e-4
GAS_CONST = 8.3144621 * J / mol
B_AV = 1e-44 * kg * m ** 2


RESTART_FILE = "restart.pickle"
OPT_KEYS = ["steps", "fmax"]
MAX_ROUNDS = 20
NUM_CONFS = 20
OPT_FILENAME = "opt.traj"
DEFAULT_INFO_FILE = "job_info.json"

INTEGRATOR_DIC = {"velocityverlet": VelocityVerlet}

CM_2_AU = 4.5564e-6
ANGS_2_AU = 1.8897259886
AMU_2_AU = 1822.88985136
k_B = 1.38064852e-23
PLANCKS_CONS = 6.62607015e-34


def get_key(iroot, num_states):
    """
    Get energy key for the state of interest.
    Args:
        iroot (int): state of interest
        num_states (int): total number of states
    Returns:
        key (str): energy key
    """

    # energy if only one state
    if iroot == 0 and num_states == 1:
        key = "energy"

    # otherwise energy with state suffix
    else:
        key = "energy_{}".format(iroot)
    return key


def init_calculator(atoms, params):
    """
    Set the calculator for the atoms and
    get the model.
    Args:
        atoms (AtomsBatch): atoms for geom of interest
        params (dict): dictionary of parameters
    Returns:
        model (nn.Module): nnpotential model
        en_key (str): energy key 
    """

    opt_state = params.get("iroot", 0)
    num_states = params.get("num_states", 1)
    en_key = get_key(iroot=opt_state, num_states=num_states)

    nn_id = params['nnid']
    # get the right weightpath (either regular or cluster-mounted)
    # depending on which exists
    weightpath = os.path.join(params['weightpath'], str(nn_id))
    if not os.path.isdir(weightpath):
        weightpath = os.path.join(params['mounted_weightpath'], str(nn_id))

    # get the model
    nn_params = params.get("networkhyperparams", {})
    model_type = params.get("model_type")
    model = load_model(weightpath,
                       model_type=model_type,
                       params=nn_params)

    # get and set the calculator
    nff_ase = NeuralFF.from_file(
        weightpath,
        device=params.get('device', 'cuda'),
        output_keys=[en_key],
        params=nn_params,
        model_type=model_type,
        needs_angles=params.get("needs_angles", False),
    )

    atoms.set_calculator(nff_ase)

    return model, en_key


def correct_hessian(restart_file, hessian):
    """
    During an optimization, replace the approximate BFGS
    Hessian with the analytical nnpotential Hessian.
    Args:
        restart_file (str): name of the pickle file
            for restarting the optimization.
        hessian (list): analytical Hessian
    Returns:
        None
    """

    # get the parameters from the restart file

    with open(restart_file, "rb") as f:
        restart = pickle.load(f)

    # set the Hessian with ase units

    hess = np.array(hessian) * units.Hartree / (units.Bohr) ** 2
    restart = tuple([hess] + restart[1:])

    # save the restart file

    with open(restart_file, "wb") as f:
        pickle.dump(restart, f)


def get_output_keys(model):

    atomwisereadout = model.atomwisereadout
    # get the names of all the attributes of the readout dict
    readout_attr_names = dir(atomwisereadout)

    # restrict to the attributes that are ModuleDicts
    readout_dict_names = [name for name in readout_attr_names if
                          type(getattr(atomwisereadout, name)) is ModuleDict]

    # get the ModuleDicts
    readout_dicts = [getattr(atomwisereadout, name)
                     for name in readout_dict_names]

    # get their keys
    output_keys = [key for dic in readout_dicts for key in dic.keys()]

    return output_keys


def get_loader(model,
               nxyz_list,
               num_states,
               cutoff,
               needs_angles=False,
               base_keys=['energy']):

    # base_keys = get_output_keys(model)
    grad_keys = [key + "_grad" for key in base_keys]

    ref_quant = [0] * len(nxyz_list)
    ref_quant_grad = [
        np.zeros(((len(nxyz_list[0])), 3)).tolist()] * len(nxyz_list)

    props = {"nxyz": nxyz_list}
    props.update({key: ref_quant for key in base_keys})
    props.update({key: ref_quant_grad for key in grad_keys})

    dataset = Dataset(props.copy())
    dataset.generate_neighbor_list(cutoff)
    if needs_angles:
        dataset.generate_angle_list()

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_dicts)

    return model, loader


def check_convg(model, loader, energy_key, device, restart_file):

    mode_dic = get_modes(model=model,
                         loader=loader,
                         energy_key=energy_key,
                         device=device)

    freqs = mode_dic["freqs"]
    neg_freqs = list(filter(lambda x: x < 0, freqs))
    num_neg = len(neg_freqs)
    if num_neg != 0:
        print(("Found {} negative frequencies; "
               "restarting optimization.").format(num_neg))
        correct_hessian(restart_file=restart_file, hessian=mode_dic["hess"])
        return False, mode_dic
    else:
        print(("Found no negative frequencies; "
               "optimization complete."))

        return True, mode_dic


def get_opt_kwargs(params):

    # params with the right name for max_step
    new_params = copy.deepcopy(params)
    new_params["steps"] = new_params["opt_max_step"]
    new_params.pop("opt_max_step")

    opt_kwargs = {key: val for key,
                  val in new_params.items() if key in OPT_KEYS}

    return opt_kwargs


def opt_conformer(atoms, params):

    converged = False
    device = params.get("device", "cuda")
    restart_file = params.get("restart_file", RESTART_FILE)
    num_states = params.get("num_states", 1)
    cutoff = params.get("cutoff", 5)
    max_rounds = params.get("max_rounds", MAX_ROUNDS)

    nn_params = params.get("networkhyperparams", {})
    output_keys = nn_params.get("output_keys", ["energy"])

    for iteration in tqdm(range(max_rounds)):

        model, energy_key = init_calculator(atoms=atoms, params=params)

        opt_module = getattr(optimize, params.get("opt_type", "BFGS"))
        opt_kwargs = get_opt_kwargs(params)
        dyn = opt_module(atoms, restart=restart_file)
        dyn_converged = dyn.run(**opt_kwargs)

        nxyz_list = [atoms.get_nxyz()]

        model, loader = get_loader(model=model,
                                   nxyz_list=nxyz_list,
                                   num_states=num_states,
                                   cutoff=cutoff,
                                   needs_angles=params.get(
                                       "needs_angles", False),
                                   base_keys=output_keys)

        hess_converged, mode_dic = check_convg(model=model,
                                               loader=loader,
                                               energy_key=energy_key,
                                               device=device,
                                               restart_file=restart_file)
        if dyn_converged and hess_converged:
            converged = True
            break

    return atoms, converged, mode_dic


def get_confs(traj_filename, thermo_filename, num_starting_poses):

    with open(thermo_filename, "r") as f:
        lines = f.readlines()
    energies = []
    for line in lines:
        try:
            energies.append(float(line.split()[2]))
        except ValueError:
            pass

    sort_idx = np.argsort(energies)
    sorted_steps = np.array(range(len(lines)))[sort_idx[:num_starting_poses]]

    trj = AseTrajectory(traj_filename)
    best_confs = [AtomsBatch(trj[i]) for i in sorted_steps]

    return best_confs


def get_nve_params(params):
    nve_params = copy.deepcopy(nve.DEFAULTNVEPARAMS)
    common_keys = [key for key in nve_params.keys() if key in params]
    for key in common_keys:
        nve_params[key] = params[key]

    integrator = nve_params["thermostat"]
    if type(integrator) is str:
        integ_name = integrator.lower().replace("_", "")
        nve_params["integrator"] = INTEGRATOR_DIC[integ_name]

    return nve_params


def md_to_conf(params):

    thermo_filename = params.get(
        "thermo_filename", nve.DEFAULTNVEPARAMS["thermo_filename"])
    if os.path.isfile(thermo_filename):
        os.remove(thermo_filename)

    nve_params = get_nve_params(params)
    nxyz = np.array(params['nxyz'])
    atoms = AtomsBatch(nxyz[:, 0], nxyz[:, 1:])
    _, _ = init_calculator(atoms=atoms, params=params)
    nve_instance = nve.Dynamics(atomsbatch=atoms,
                                mdparam=nve_params)
    nve_instance.run()

    thermo_filename = params.get(
        "thermo_filename", nve.DEFAULTNVEPARAMS["thermo_filename"])
    traj_filename = params.get(
        "traj_filename", nve.DEFAULTNVEPARAMS["traj_filename"])

    num_starting_poses = params.get("num_starting_poses", NUM_CONFS)
    best_confs = get_confs(traj_filename=traj_filename,
                           thermo_filename=thermo_filename,
                           num_starting_poses=num_starting_poses)

    return best_confs


def confs_to_opt(params, best_confs):

    convg_atoms = []
    energy_list = []
    mode_list = []

    for i in range(len(best_confs)):
        atoms = best_confs[i]
        atoms, converged, mode_dic = opt_conformer(atoms=atoms, params=params)

        if converged:
            convg_atoms.append(atoms)
            energy_list.append(atoms.get_potential_energy())
            mode_list.append(mode_dic)

    if not convg_atoms:
        raise Exception("No successful optimizations")

    # sort results by energy
    best_idx = np.argsort(np.array(energy_list)).reshape(-1)
    best_atoms = [convg_atoms[i] for i in best_idx]
    best_modes = [mode_list[i] for i in best_idx]

    return best_atoms, best_modes


def get_opt_and_modes(params):

    best_confs = md_to_conf(params)
    all_geoms, all_modes = confs_to_opt(params=params,
                                        best_confs=best_confs)
    opt_geom = all_geoms[0]
    mode_dic = all_modes[0]

    return opt_geom, mode_dic


def get_orca_form(cc_mat, cc_freqs, n_atoms):
    """ Converts cclib version of Orca's (almost orthogonalizing) matrix 
    and mode frequencies back into the original
    Orca forms. Also converts frequencies from cm^{-1}
     into atomic units (Hartree)."""

    pure_matrix = np.asarray(cc_mat)
    pure_freqs = np.asarray(cc_freqs)
    n_modes = len(pure_matrix[:, 0])
    n_inactive = n_atoms*3 - len(pure_matrix[:, 0])
    n_tot = n_modes + n_inactive

    for i in range(len(pure_matrix)):

        new_col = pure_matrix[i].reshape(3*len(pure_matrix[i]))
        if i == 1:
            new_mat = np.column_stack((old_col, new_col))
        elif i > 1:
            new_mat = np.column_stack((new_mat, new_col))
        old_col = new_col[:]

    matrix = np.asarray(new_mat[:]).reshape(n_tot, n_modes)

    zero_col = np.asarray([[0]]*len(matrix))
    for i in range(0, n_inactive):
        matrix = np.insert(matrix, [0], zero_col, axis=1)
    freqs = np.asarray(pure_freqs[:])
    for i in range(0, n_inactive):
        freqs = np.insert(freqs, 0, 0)

    return matrix, freqs * CM_2_AU


def get_orth(mass_vec, matrix):
    """Makes orthogonalizing matrix given the outputted 
        (non-orthogonal) matrix from Orca. The mass_vec variable 
        is a list of the masses of the atoms in the molecule (must be)
        in the order given to Orca when it calculated normal modes).
       Note that this acts directly on the matrix outputted from Orca,
       not on the cclib version that divides columns into sets of 
       three entries for each atom."""

    m = np.array([[mass] for mass in mass_vec])
    # repeat sqrt(m) three times, one for each direction
    sqrt_m_vec = np.kron(m ** 0.5, np.ones((3, 1)))
    # a matrix with sqrt_m repeated K times, where
    # K = 3N - 5 or 3N-6 is the number of modes
    sqrt_m_mat = np.kron(sqrt_m_vec, np.ones(
        (1, len(sqrt_m_vec))))

    # orthogonalize the matrix by element-wise multiplication with 1/sqrt(m)
    orth = sqrt_m_mat * matrix

    for i in range(len(orth)):
        if np.linalg.norm(orth[:, i]) != 0:
            # normalize the columns
            orth[:, i] = orth[:, i] / np.linalg.norm(orth[:, i])

    return orth, np.reshape(sqrt_m_vec, len(sqrt_m_vec))


def get_n_in(matrix):
    """ Get number of inactive modes """

    n_in = 0
    for entry in matrix[0]:
        if entry == 0:
            n_in += 1
    return n_in


def get_disp(mass_vec, matrix, freqs, q, p, hb=1):
    """Makes position and momentum displacements from 
    unitless harmonic oscillator displacements and unitless momenta.
    Uses atomic units (hbar = 1). For different units change the value of hbar."""

    orth, sqrt_m_vec = get_orth(mass_vec, matrix)
    n_in = get_n_in(matrix)

    # get actual positions dq from unitless positions q

    q_tilde = q[n_in:] * (hb / (freqs[n_in:])) ** 0.5
    q_tilde = np.append(np.zeros(n_in), q_tilde)
    # multiply by orth, divide element-wise by sqrt(m)
    dq = np.matmul(orth, q_tilde) / sqrt_m_vec

    # get actual momenta p_tilde from unitless momenta p

    p_tilde = p * (hb * (freqs)) ** 0.5
    dp = np.matmul(orth, p_tilde) * sqrt_m_vec

    return dq, dp


def wigner_sample(w, kt=25.7 / 1000 / 27.2, hb=1):
    """ Sample unitless x and unitless p from a Wigner distribution. 
    Takes frequency and temperature in au as inputs.
    Default temperature is 300 K."""

    sigma = (1/np.tanh((hb*w)/(2*kt)))**0.5/2**0.5
    cov = [[sigma**2, 0], [0, sigma**2]]
    mean = (0, 0)
    x, p = np.random.multivariate_normal(mean, cov)
    return x, p


def classical_sample(w,  kt=25.7 / 1000 / 27.2, hb=1):
    sigma = (kt / (hb * w)) ** 0.5
    cov = [[sigma**2, 0], [0, sigma**2]]
    mean = (0, 0)
    x, p = np.random.multivariate_normal(mean, cov)
    return x, p


def make_dx_dp(mass_vec,
               cc_matrix,
               cc_freqs,
               kt=25.7 / 1000 / 27.2,
               hb=1,
               classical=False):
    """Make Wigner-sampled p and dx, where dx is the displacement
     about the equilibrium geometry.
    Takes mass vector, CClib matrix, and CClib vib freqs as inputs.
    Inputs in au unless hb is specified in different coordinates. Default 300 K."""

    matrix, freqs = get_orca_form(cc_matrix, cc_freqs, n_atoms=len(mass_vec))
    unitless_x = np.array([])
    unitless_p = np.array([])
    n_in = get_n_in(matrix)

    for w in freqs[n_in:]:
        if classical:
            x, p = classical_sample(w, kt, hb=hb)
        else:
            x, p = wigner_sample(w, kt, hb=hb)
        unitless_x = np.append(unitless_x, x)
        unitless_p = np.append(unitless_p, p)

    unitless_x = np.append(np.zeros(n_in), unitless_x)
    unitless_p = np.append(np.zeros(n_in), unitless_p)

    dx, dp = get_disp(mass_vec=mass_vec,
                      matrix=matrix,
                      freqs=freqs,
                      q=unitless_x,
                      p=unitless_p,
                      hb=hb)

    # re-shape to have form of [[dx1, dy1, dz1], [dx2, dy2, dz2], ...]

    n_atoms = int(len(dx) / 3)
    shaped_dx, shaped_dp = dx.reshape(n_atoms, 3), dp.reshape(n_atoms, 3)

    return shaped_dx, shaped_dp


def split_convert_xyz(xyz):
    """ Splits xyz into Z, coordinates in au, and masses in au """
    coords = [(np.array(element[1:])*ANGS_2_AU).tolist() for element in xyz]
    mass_vec = [PERIODICTABLE.GetAtomicWeight(
        int(element[0]))*AMU_2_AU for element in xyz]
    Z = [element[0] for element in xyz]
    return Z, coords, mass_vec


def join_xyz(Z, coords):
    """ Joins Z's and coordinates back into xyz """
    out = []
    for i in range(len(coords)):
        this_quad = [Z[i]]
        this_quad += coords[i]
        out.append(this_quad)


def make_wigner_init(init_atoms,
                     vibdisps,
                     vibfreqs,
                     num_samples,
                     kt=25.7 / 1000 / 27.2,
                     hb=1,
                     classical=False):
    """Generates Wigner-sampled coordinates and velocities. 
    xyz is the xyz array at the optimized
    geometry. xyz is in Angstrom, so xyz is first converted to 
    au, added to Wigner dx, and then
    converted back to Angstrom. Velocity is in au. 
    vibdisps and vibfreqs are the CClib quantities
    found in the database."""

    xyz = np.concatenate([init_atoms.get_atomic_numbers().reshape(-1, 1),
                          init_atoms.get_positions()], axis=1)
    atoms_list = []

    for _ in range(num_samples):
        assert min(
            vibfreqs) >= 0, ("Negative frequencies found. "
                             "Geometry must not be converged.")

        Z, opt_coords, mass_vec = split_convert_xyz(xyz)
        dx, dp = make_dx_dp(mass_vec, vibdisps, vibfreqs,
                            kt, hb, classical=classical)
        wigner_coords = ((np.asarray(opt_coords) + dx)/ANGS_2_AU).tolist()

        nxyz = np.array(join_xyz(Z, wigner_coords))
        velocity = (dp / np.array([[m] for m in mass_vec])).tolist()

        atoms = AtomsBatch(nxyz[:, 0], nxyz[:, 1:])

        # conv = EV_TO_AU / (ASE_TO_FS * FS_TO_AU)
        conv = 1 / BOHR_RADIUS / (ASE_TO_FS * FS_TO_AU)
        atoms.set_velocities(np.array(velocity) / conv)

        atoms_list.append(atoms)

    return atoms_list


def nms_sample(params,
               classical,
               num_samples,
               kt=25.7 / 1000 / 27.2,
               hb=1):

    atoms, mode_dic = get_opt_and_modes(params)
    vibdisps = np.array(mode_dic["modes"])
    vibdisps = vibdisps.reshape(vibdisps.shape[0], -1, 3).tolist()

    vibfreqs = mode_dic["freqs"]

    atoms_list = make_wigner_init(init_atoms=atoms,
                                  vibdisps=vibdisps,
                                  vibfreqs=vibfreqs,
                                  num_samples=num_samples,
                                  kt=kt,
                                  hb=hb,
                                  classical=classical)

    return atoms_list


def get_modes(model, loader, energy_key, device):

    batch = next(iter(loader))
    batch = batch_to(batch, device)
    model = model.to(device)

    if isinstance(model, SchNet):
        hessian = get_schnet_hessians(batch=batch,
                                      model=model,
                                      device=device,
                                      energy_key=energy_key)[
            0].cpu().detach().numpy()
    else:
        raise NotImplementedError

    # convert to Ha / bohr^2
    hessian *= (const.BOHR_RADIUS) ** 2
    hessian *= const.KCAL_TO_AU['energy']

    force_consts, vib_freqs, eigvec = vib_analy(
        r=batch["nxyz"][:, 0].cpu().detach().numpy(),
        xyz=batch["nxyz"][:, 1:].cpu().detach().numpy(),
        hessian=hessian)

    # from https://gaussian.com/vib/#SECTION00036000000000000000
    nxyz = batch["nxyz"].cpu().detach().numpy()
    masses = np.array([PT.GetMostCommonIsotopeMass(int(z))
                       for z in nxyz[:, 0]])
    triple_mass = np.concatenate([np.array([item] * 3) for item in masses])
    red_mass = 1 / np.matmul(eigvec ** 2, 1 / triple_mass)

    # un-mass weight the modes
    modes = []
    for vec in eigvec:
        col = vec / triple_mass ** 0.5
        col /= np.linalg.norm(col)
        modes.append(col)
    modes = np.array(modes)

    out_dic = {"nxyz": nxyz.tolist(),
               "hess": hessian.tolist(),
               "modes": modes.tolist(),
               "red_mass": red_mass.tolist(),
               "freqs": vib_freqs.tolist()}

    return out_dic


def moi_tensor(massvec, expmassvec, xyz):
    # Center of Mass
    com = np.sum(expmassvec.reshape(-1, 3) *
                 xyz.reshape(-1, 3), axis=0
                 ) / np.sum(massvec)

    # xyz shifted to COM
    xyz_com = xyz.reshape(-1, 3) - com

    # Compute elements need to calculate MOI tensor
    mass_xyz_com_sq_sum = np.sum(
        expmassvec.reshape(-1, 3) * xyz_com ** 2, axis=0)

    mass_xy = np.sum(massvec * xyz_com[:, 0] * xyz_com[:, 1], axis=0)
    mass_yz = np.sum(massvec * xyz_com[:, 1] * xyz_com[:, 2], axis=0)
    mass_xz = np.sum(massvec * xyz_com[:, 0] * xyz_com[:, 2], axis=0)

    # MOI tensor
    moi = np.array([[mass_xyz_com_sq_sum[1] + mass_xyz_com_sq_sum[2], -1 *
                     mass_xy, -1 * mass_xz],
                    [-1 * mass_xy, mass_xyz_com_sq_sum[0] +
                        mass_xyz_com_sq_sum[2], -1 * mass_yz],
                    [-1 * mass_xz, -1 * mass_yz, mass_xyz_com_sq_sum[0] +
                     mass_xyz_com_sq_sum[1]]])

    # MOI eigenvectors and eigenvalues
    moi_eigval, moi_eigvec = np.linalg.eig(moi)

    return xyz_com, moi_eigvec


def trans_rot_vec(massvec, xyz_com, moi_eigvec):

    # Mass-weighted translational vectors
    zero_vec = np.zeros([len(massvec)])
    sqrtmassvec = np.sqrt(massvec)
    expsqrtmassvec = np.repeat(sqrtmassvec, 3)

    d1 = np.transpose(np.stack((sqrtmassvec, zero_vec, zero_vec))).reshape(-1)
    d2 = np.transpose(np.stack((zero_vec, sqrtmassvec, zero_vec))).reshape(-1)
    d3 = np.transpose(np.stack((zero_vec, zero_vec, sqrtmassvec))).reshape(-1)

    # Mass-weighted rotational vectors
    big_p = np.matmul(xyz_com, moi_eigvec)

    d4 = (np.repeat(big_p[:, 1], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 2], len(massvec)).reshape(-1) -
          np.repeat(big_p[:, 2], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 1], len(massvec)).reshape(-1)
          ) * expsqrtmassvec

    d5 = (np.repeat(big_p[:, 2], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 0], len(massvec)).reshape(-1) -
          np.repeat(big_p[:, 0], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 2], len(massvec)).reshape(-1)
          ) * expsqrtmassvec

    d6 = (np.repeat(big_p[:, 0], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 1], len(massvec)).reshape(-1) -
          np.repeat(big_p[:, 1], 3).reshape(-1) *
          np.tile(moi_eigvec[:, 0], len(massvec)).reshape(-1)
          ) * expsqrtmassvec

    d1_norm = d1 / np.linalg.norm(d1)
    d2_norm = d2 / np.linalg.norm(d2)
    d3_norm = d3 / np.linalg.norm(d3)
    d4_norm = d4 / np.linalg.norm(d4)
    d5_norm = d5 / np.linalg.norm(d5)
    d6_norm = d6 / np.linalg.norm(d6)

    dx_norms = np.stack((d1_norm,
                         d2_norm,
                         d3_norm,
                         d4_norm,
                         d5_norm,
                         d6_norm))

    return dx_norms


def vib_analy(r, xyz, hessian):

    # r is the proton number of atoms
    # xyz is the cartesian coordinates in Angstrom
    # Hessian elements in atomic units (Ha/bohr^2)

    massvec = np.array([PT.GetAtomicWeight(i.item()) * AMU2KG
                        for i in list(np.array(r.reshape(-1)).astype(int))])
    expmassvec = np.repeat(massvec, 3)
    sqrtinvmassvec = np.divide(1.0, np.sqrt(expmassvec))
    hessian_mwc = np.einsum('i,ij,j->ij', sqrtinvmassvec,
                            hessian, sqrtinvmassvec)
    hessian_eigval, hessian_eigvec = np.linalg.eig(hessian_mwc)

    xyz_com, moi_eigvec = moi_tensor(massvec, expmassvec, xyz)
    dx_norms = trans_rot_vec(massvec, xyz_com, moi_eigvec)

    P = np.identity(3 * len(massvec))
    for dx_norm in dx_norms:
        P -= np.outer(dx_norm, dx_norm)

    # Projecting the T and R modes out of the hessian
    mwhess_proj = np.dot(P.T, hessian_mwc).dot(P)
    hess_proj = np.einsum('i,ij,j->ij', 1 / sqrtinvmassvec,
                          mwhess_proj, 1 / sqrtinvmassvec)

    hessian_eigval, hessian_eigvec = np.linalg.eigh(mwhess_proj)

    neg_ele = []
    for i, eigval in enumerate(hessian_eigval):
        if eigval < 0:
            neg_ele.append(i)

    hessian_eigval_abs = np.abs(hessian_eigval)

    pre_vib_freq_cm_1 = np.sqrt(
        hessian_eigval_abs * HA2J * 10e19) / (SPEEDOFLIGHT * 2 * np.pi *
                                              BOHRS2ANG * 100)

    vib_freq_cm_1 = pre_vib_freq_cm_1.copy()

    for i in neg_ele:
        vib_freq_cm_1[i] = -1 * pre_vib_freq_cm_1[i]

    trans_rot_elms = []
    for i, freq in enumerate(vib_freq_cm_1):
        # Modes that are less than 1.0 cm-1 are the
        # translation / rotation modes we just projected
        # out
        if np.abs(freq) < 1.0:
            trans_rot_elms.append(i)

    force_constants_J_m_2 = np.delete(
        hessian_eigval * HA2J * 1e20 / (BOHRS2ANG ** 2) * AMU2KG,
        trans_rot_elms)

    proj_vib_freq_cm_1 = np.delete(vib_freq_cm_1, trans_rot_elms)
    proj_hessian_eigvec = np.delete(hessian_eigvec.T, trans_rot_elms, 0)

    return (force_constants_J_m_2, proj_vib_freq_cm_1, proj_hessian_eigvec,
            mwhess_proj, hess_proj)


def free_rotor_moi(freqs):
    freq_ev = freqs * CM_TO_EV
    mu = 1 / (8 * np.pi ** 2 * freq_ev)
    return mu


def eff_moi(mu, b_av):
    mu_prime = mu * b_av / (mu + b_av)
    return mu_prime


def low_freq_entropy(freqs,
                     temperature,
                     b_av=B_AV):
    mu = free_rotor_moi(freqs)
    mu_prime = eff_moi(mu, b_av)

    arg = (8 * np.pi ** 3 * mu_prime * kB * temperature)
    entropy = GAS_CONST * (1 / 2 + np.log(arg ** 0.5))

    return entropy


def high_freq_entropy(freqs,
                      temperature):

    freq_ev = freqs * CM_TO_EV
    exp_pos = np.exp(freq_ev / (kB * temperature)) - 1
    exp_neg = 1 - np.exp(-freq_ev / (kB * temperature))

    entropy = GAS_CONST * (
        freq_ev / (kB * temperature * exp_pos) -
        np.log(exp_neg)
    )

    return entropy


def mrrho_entropy(freqs,
                  temperature,
                  rotor_cutoff,
                  b_av,
                  alpha):

    func = 1 / (1 + (rotor_cutoff / freqs) ** alpha)
    s_r = low_freq_entropy(freqs=freqs,
                           b_av=b_av,
                           temperature=temperature)
    s_v = high_freq_entropy(freqs=freqs,
                            temperature=temperature)

    new_vib_s = (func * s_v + (1 - func) * s_r).sum()
    old_vib_s = s_v.sum()

    return old_vib_s, new_vib_s


def mrrho_quants(ase_atoms,
                 freqs,
                 imag_cutoff=IMAG_CUTOFF,
                 temperature=TEMP,
                 pressure=PRESSURE,
                 rotor_cutoff=ROTOR_CUTOFF,
                 b_av=B_AV,
                 alpha=4,
                 flip_all_but_ts=False):

    potentialenergy = ase_atoms.get_potential_energy()

    if flip_all_but_ts:
        print(("Flipping all imaginary frequencies except "
               "the lowest one"))
        abs_freqs = abs(freqs[1:])

    else:
        abs_freqs = abs(freqs[freqs > imag_cutoff])
    ens = abs_freqs * CM_TO_EV

    ideal_gas = IdealGasThermo(vib_energies=ens,
                               potentialenergy=potentialenergy,
                               atoms=ase_atoms,
                               geometry='nonlinear',
                               symmetrynumber=1,
                               spin=0)

    # full entropy including rotation, translation etc
    old_entropy = (ideal_gas.get_entropy(temperature=temperature,
                                         pressure=pressure).item())
    enthalpy = (ideal_gas.get_enthalpy(temperature=temperature)
                .item())

    # correction to vibrational entropy
    out = mrrho_entropy(freqs=abs_freqs,
                        temperature=temperature,
                        rotor_cutoff=rotor_cutoff,
                        b_av=b_av,
                        alpha=alpha)
    old_vib_s, new_vib_s = out
    final_entropy = old_entropy - old_vib_s + new_vib_s

    free_energy = (enthalpy - temperature * final_entropy)

    return final_entropy, enthalpy, free_energy


def convert_modes(atoms,
                  modes):

    masses = (atoms.get_masses().reshape(-1, 1)
              .repeat(3, 1)
              .reshape(1, -1))

    # Multiply by 1 / sqrt(M) to be consistent with the DB
    vibdisps = modes / (masses ** 0.5)
    norm = np.linalg.norm(vibdisps, axis=1).reshape(-1, 1)

    # Normalize
    vibdisps /= norm

    # Re-shape

    num_atoms = len(atoms)
    vibdisps = vibdisps.reshape(-1, num_atoms, 3)

    return vibdisps


def hessian_and_modes(ase_atoms,
                      imag_cutoff=IMAG_CUTOFF,
                      rotor_cutoff=ROTOR_CUTOFF,
                      temperature=TEMP,
                      pressure=PRESSURE,
                      flip_all_but_ts=False,
                      analytical=False):

    # comparison to the analytical Hessian
    # shows that delta=0.005 is indistinguishable
    # from the real result, whereas delta=0.05
    # has up to 20% errors

    # delete the folder `vib` if it exists,
    # because it might mess up the Hessian
    # calculation

    if os.path.isdir('vib'):
        shutil.rmtree('vib')

    if analytical:
        hessian = analytical_hess(atoms=ase_atoms)

    else:
        vib = Vibrations(ase_atoms, delta=0.005)
        vib.run()

        vib_results = vib.get_vibrations()
        dim = len(ase_atoms)
        hessian = (vib_results.get_hessian()
                   .reshape(dim * 3, dim * 3) *
                   EV_TO_AU *
                   BOHR_RADIUS ** 2)

        print(vib.get_frequencies()[:20])

    vib_results = vib_analy(r=ase_atoms.get_atomic_numbers(),
                            xyz=ase_atoms.get_positions(),
                            hessian=hessian)
    _, freqs, modes, mwhess_proj, hess_proj = vib_results
    mwhess_proj *= AMU2KG

    vibdisps = convert_modes(atoms=ase_atoms,
                             modes=modes)

    mrrho_results = mrrho_quants(ase_atoms=ase_atoms,
                                 freqs=freqs,
                                 imag_cutoff=imag_cutoff,
                                 temperature=temperature,
                                 pressure=pressure,
                                 rotor_cutoff=rotor_cutoff,
                                 flip_all_but_ts=flip_all_but_ts)

    entropy, enthalpy, free_energy = mrrho_results

    imgfreq = len(freqs[freqs < 0])
    results = {"vibdisps": vibdisps.tolist(),
               "vibfreqs": freqs.tolist(),
               "modes": modes,
               "hessianmatrix": hessian.tolist(),
               "mwhess_proj": mwhess_proj.tolist(),
               "hess_proj": hess_proj.tolist(),
               "imgfreq": imgfreq,
               "freeenergy": free_energy * EV_TO_AU,
               "enthalpy": enthalpy * EV_TO_AU,
               "entropy": entropy * temperature * EV_TO_AU}

    return results
