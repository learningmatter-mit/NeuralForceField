import numpy as np
import os
from torch.utils.data import DataLoader
from torch.nn.modules.container import ModuleDict
import copy
import pickle
from rdkit import Chem

from ase import optimize, units
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory as AseTrajectory


from nff.io.ase_ax import NeuralFF, AtomsBatch
from nff.train import load_model
from nff.data import collate_dicts, Dataset
from nff.md import nve
from neuralnet.utils.vib import get_modes

from tqdm import tqdm


from nff.utils.constants import FS_TO_AU, ASE_TO_FS, EV_TO_AU, BOHR_RADIUS

PERIODICTABLE = Chem.GetPeriodicTable()

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
HA2J = 4.359744E-18
BOHRS2ANG = 0.529177
SPEEDOFLIGHT = 2.99792458E8
AMU2KG = 1.660538782E-27


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
    restart = (hess, *restart[1:])

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

    props = {"nxyz": nxyz_list, **{key: ref_quant for key in base_keys},
             **{key: ref_quant_grad for key in grad_keys}}

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
    return [[Z[i], *coords[i]] for i in range(len(coords))]


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
        conv =  1 / BOHR_RADIUS / (ASE_TO_FS * FS_TO_AU)
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


if __name__ == "__main__":

    true = True
    params = {"htvs": "$HOME/htvs",
              "T_init": 300.0,
              "temp": 300,
              "time_step": 0.5,
              "integrator": "velocity_verlet",
              "steps": 2,
              "save_frequency": 1,
              "nbr_list_update_freq": 1,
              "thermo_filename": "./thermo.log",
              "traj_filename": "./atom.traj",
              "num_states": 2,
              "iroot": 0,
              "opt_max_step": 2000,
              "fmax": 0.05,
              # "fmax": 10,
              "hess_interval": 100000,
              "num_starting_poses": 1,
              "method": {"name": "sf_tddft_bhhlyp",
                         "description": ("GAMESS bhhlyp/6-31G*"
                                         " spin flip tddft")},
              "num_save": 1,
              "nms": true,
              "classical": true,
              "weightpath": "/home/saxelrod/models",
              "nnid": "azo_dimenet_diabat",
              "networkhyperparams": {
                  "n_rbf": 6,
                  "cutoff": 5.0,
                  "envelope_p": 5,
                  "n_spher": 7,
                  "l_spher": 7,
                  "embed_dim": 128,
                  "int_dim": 64,
                  "out_dim": 256,
                  "basis_emb_dim": 8,
                  "activation": "swish",
                  "n_convolutions": 4,
                  "use_pp": true,
                  "output_keys": ["energy_0", "energy_1"],
                  "diabat_keys": [["d0", "lam"], ["lam", "d1"]],
                  "grad_keys": ["energy_0_grad", "energy_1_grad"]
              },
              "model_type": "DimeNetDiabat",
              "needs_angles": true,
              "device": "cpu",
              "nxyz": [[6.0, -3.106523, -0.303932, 1.317003], [6.0, -2.361488, -1.070965, 0.433279], [6.0, -1.500175, -0.466757, -0.466259], [6.0, -1.394372, 0.919597, -0.489914], [7.0, -0.638906, 1.622236, -1.475649], [7.0, 0.53754, 1.376216, -1.741989], [6.0, 1.347532, 0.467035, -0.998026], [6.0, 2.132836, -0.410469, -1.735143], [6.0, 3.015726, -1.257597, -1.087982], [6.0, 3.157147, -1.193886, 0.290324], [6.0, 2.407287, -0.281797, 1.018438], [6.0, 1.497631, 0.545173, 0.382205], [6.0, -2.176213, 1.69203, 0.35984], [6.0, -3.00977, 1.079171, 1.279395], [1.0, -3.769581, -0.781063, 2.019916], [1.0, -2.44819, -2.145275, 0.44532], [1.0, -0.921598, -1.062715, -1.15082], [1.0, 2.038044, -0.418366, -2.808393], [1.0, 3.607298, -1.952544, -1.661382], [1.0, 3.858162, -1.840041, 0.792768], [1.0, 2.528111, -0.214815, 2.087415], [1.0, 0.915297, 1.251752, 0.948046], [1.0, -2.117555, 2.765591, 0.289571], [1.0, -3.598241, 1.681622, 1.952038]]

              }

    try:
        atoms_list = nms_sample(params=params,
                                classical=True,
                                num_samples=10,
                                kt=25.7 / 1000 / 27.2,
                                hb=1)
    except Exception as e:
        print(e)
        import pdb
        pdb.post_mortem()
