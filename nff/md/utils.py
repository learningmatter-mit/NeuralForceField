
import csv
import copy

from ase import units
from ase.md import MDLogger


import nff.utils.constants as const


class NeuralMDLogger(MDLogger):
    def __init__(self,
                 *args,
                 verbose=True,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.verbose = verbose
        if verbose:
            print(self.hdr)

        self.natoms = len(self.atoms)

    def __call__(self):
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * self.natoms)
        if self.peratom:
            epot /= self.natoms
            ekin /= self.natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000*units.fs)
            dat = (t,)
        else:
            dat = ()
        if not isinstance(epot, float):
            ekin = ekin/len(epot)
            epot = sum(epot)/len(epot)
        dat += (epot+ekin, epot, ekin, temp)
        if self.stress:
            dat += tuple(self.atoms.get_stress() / units.GPa)
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()

        if self.verbose:
            print(self.fmt % dat)


def get_energy(atoms):
    """Function to print the potential, kinetic and total energy"""
    epot = atoms.get_potential_energy() * const.EV_TO_KCAL_MOL  # / len(atoms)
    ekin = atoms.get_kinetic_energy() * const.EV_TO_KCAL_MOL  # / len(atoms)
    Temperature = ekin / (1.5 * units.kB * len(atoms))

    # compute kinetic energy by hand
    # vel = torch.Tensor(atoms.get_velocities())
    # mass = atoms.get_masses()
    # mass = torch.Tensor(mass)
    # ekin = (0.5 * (vel * 1e-10 * fs * 1e15).pow(2).sum(1) * (mass * 1.66053904e-27) * 6.241509e+18).sum()
    # ekin = ekin.item() #* ev_to_kcal

    #ekin = ekin.detach().numpy()

    print('Energy per atom: Epot = %.2fkcal/mol  Ekin = %.2fkcal/mol (T=%3.0fK)  '
          'Etot = %.2fkcal/mol' % (epot, ekin, Temperature, epot + ekin))
    # print('Energy per atom: Epot = %.5feV  Ekin = %.5feV (T=%3.0fK)  '
    #      'Etot = %.5feV' % (epot, ekin, Temperature, (epot + ekin)))
    return epot, ekin, Temperature


def write_traj(filename, frames):
    '''
        Write trajectory dataframes into .xyz format for VMD visualization
        to do: include multiple atom types 

        example:
            path = "../../sim/topotools_ethane/ethane-nvt_unwrap.xyz"
            traj2write = trajconv(n_mol, n_atom, box_len, path)
            write_traj(path, traj2write)
    '''
    file = open(filename, 'w')
    atom_no = frames.shape[1]
    for i, frame in enumerate(frames):
        file.write(str(atom_no) + '\n')
        file.write('Atoms. Timestep: ' + str(i)+'\n')
        for atom in frame:
            if atom.shape[0] == 4:
                try:
                    file.write(str(int(atom[0])) + " " + str(atom[1]) +
                               " " + str(atom[2]) + " " + str(atom[3]) + "\n")
                except:
                    file.write(str(atom[0]) + " " + str(atom[1]) +
                               " " + str(atom[2]) + " " + str(atom[3]) + "\n")
            elif atom.shape[0] == 3:
                file.write(
                    "1" + " " + str(atom[0]) + " " + str(atom[1]) + " " + str(atom[2]) + "\n")
            else:
                raise ValueError("wrong format")
    file.close()


def csv_read(out_file):
    """
    Read a csv output file.
    Args:
        out_file (str): name of output file
    Returns:
        dic_list (list): list of dictionaries
    """

    with open(out_file, newline='') as csvfile:
        # get the keys and the corresponding dictionaries
        # being outputted
        dic_list = list(csv.DictReader(csvfile))[0::2]

    # the key ordering in the csv file may change; get `dic_keys`, the
    # dictionary that converts the key order on the first line
    # to the key order on every other line.
    # (Also, weird things happen if you define `dic_keys` and
    # `dic_list` within the same context manager, so must do it separately)
    with open(out_file, newline='') as csvfile:
        # this dictionary gives you a key: value pair
        # of the form supposed key: actual key
        dic_keys = list(csv.DictReader(csvfile))[1::2]

    # fix the key ordering
    new_dic_list = []
    for regular_dic, key_dic in zip(dic_list, dic_keys):
        new_dic = copy.deepcopy(regular_dic)
        for key in regular_dic.keys():
            new_dic[key_dic[key]] = regular_dic[key]
        new_dic_list.append(new_dic)

    for dic in new_dic_list:
        for key, value in dic.items():
            if 'nan' in value:
                value = value.replace('nan', "float('nan')")
            if 'inf' in value:
                value = value.replace('inf', "float('inf')")
            dic[key] = eval(value)

    return new_dic_list
