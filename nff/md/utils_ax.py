import os
import numpy as np
import csv
import json
import logging
import copy
import pdb


import ase
from ase import Atoms, units
from ase.md import MDLogger

from nff.utils.scatter import compute_grad
from nff.data.graphs import *
import nff.utils.constants as const


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

    print(('Energy per atom: Epot = %.2fkcal/mol  '
           'Ekin = %.2fkcal/mol (T=%3.0fK)  '
           'Etot = %.2fkcal/mol'
           % (epot, ekin, Temperature, epot + ekin)))
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
                file.write(("1" + " " + str(atom[0]) + " "
                            + str(atom[1]) + " " + str(atom[2]) + "\n"))
            else:
                raise ValueError("wrong format")
    file.close()


def mol_dot(vec1, vec2):
    """ Say we have two vectors, each of which has the form 
    [[fx1, fy1, fz1], [fx2, fy2, fz2], ...].
    mol_dot returns an array of dot products between each 
    element of the two vectors. """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    out = np.transpose([np.dot(element1, element2) for
                        element1, element2 in zip(v1, v2)])
    return out


def mol_norm(vec):
    """Square root of mol_dot(vec, vec)."""
    return mol_dot(vec, vec)**0.5


def atoms_to_nxyz(atoms, positions=None):

    atomic_numbers = atoms.get_atomic_numbers()
    if positions is None:
        positions = atoms.get_positions()

    # don't make this a numpy array or it'll become type float64,
    # which will mess up the tensor computation. Need it to be
    # type float32.
    nxyz = [[symbol, *position] for
            symbol, position in zip(atomic_numbers, positions)]

    return nxyz


def zhu_dic_to_list(dic):
    """
    Convert dictionary of items, each value of which contains the items at all time steps, to a list
    of dictionaries at each time step.
    Args:
        dic (dict): dictionary
    Returns:
        lst (list): a list of dictionaries with the proeprties of dic at each time step
    """

    lst = []
    first_key = list(dic.keys())[0]
    for i in range(len(dic[first_key])):
        sub_dic = dict()
        for key in dic.keys():
            sub_dic[key.split("_list")[0]] = dic[key][i]
            if (key == "time_list"):
                sub_dic[key.split("_list")[0]] /= const.FS_TO_AU
        lst.append(sub_dic)

    return lst


def append_to_csv(lst, out_file):
    """
    Append a list of dictionaries to a csv file.
    Args:
        out_file (str): name of output csv file
    Returns:
        None
    """
    with open(out_file, 'a+') as csvfile:
        for item in lst:
            fieldnames = sorted(list(item.keys()))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({key: item[key] for key in fieldnames})


def write_to_new_csv(lst, out_file):
    """
    Same as `append_to_csv`, but writes a new file.
    """
    with open(out_file, 'w') as csvfile:
        for item in lst:
            fieldnames = sorted(list(item.keys()))
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({key: item[key] for key in fieldnames})


def csv_write(lst, out_file, method):
    """
    Write to a csv file with either the append or write method.
    """
    assert method in ["append", "new"]
    if method == "append":
        append_to_csv(lst, out_file)
    elif method == "new":
        write_to_new_csv(lst, "temp")
        os.rename("temp", out_file)


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
            dic[key] = eval(value)

    return new_dic_list


class NeuralMDLogger(MDLogger):
    def __init__(self,
                 *args,
                 verbose=True,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.natoms = len(self.atoms)
        self.verbose = verbose

        if verbose:
            print(self.hdr)

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
        dat += (epot+ekin, epot, ekin, temp)
        if self.stress:
            dat += tuple(self.atoms.get_stress() / units.GPa)
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()

        if self.verbose:
            print(self.fmt % dat)


class ZhuNakamuraLogger:

    """
    Base class for Zhu Nakamura dynamics.
    Properties:
        out_file (str): name of output file
        log_file (str): name of log file
        save_keys (list): name of keys whose values should be saved
    """

    def __init__(self, out_file, log_file, save_keys, **kwargs):

        self.out_file = out_file
        self.log_file = log_file
        self.save_keys = save_keys

    def setup_logging(self):
        """
        Set up logging and saving for trajectory.
        """
        # remove out_file and log_file if they exist
        if os.path.exists(self.out_file):
            os.remove(self.out_file)
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    # def save_as_pymol(self):
    #     """
    #     Save trajectory in pymol format
    #     """

    #     symbols = self.atoms.get_chemical_symbols()
    #     nxyz_list = []
    #     for position, in_trj in zip(self.position_list, self.in_trj_list):
    #         if not in_trj:
    #             continue
    #         nxyz = [[symbol, *(xyz.astype("str")).tolist()] for
    #                 symbol, xyz in zip(symbols, position)]
    #         nxyz_list.append(nxyz)
    #     pymol_str = ""
    #     for xyz in nxyz_list:
    #         pymol_str += "{}\nComment\n".format(self.Natom)
    #         for sub in xyz:
    #             pymol_str += " ".join(sub) + "\n"
    #     with open(PYMOL_NAME, "w") as f:
    #         f.write(pymol_str)

    def create_save_list(self):
        """
        Get a list of the values to be saved.
        Args:
            None
        Returns:
            save_list (list): list of values
        """

        self_dic = {key: getattr(self, key) for key in self.save_keys}
        save_dic = dict()
        for key, val in self_dic.items():
            if hasattr(val[0], "tolist"):
                save_dic[key] = [sub_val.tolist() for sub_val in val]
            else:
                save_dic[key] = val

        save_dic["nxyz_list"] = [atoms_to_nxyz(self.atoms, positions) for
                                 positions in save_dic["position_list"]]
        save_list = zhu_dic_to_list(save_dic)

        return save_list

    def save(self):
        """
        Save values
        """

        save_list = self.create_save_list()
        csv_write(out_file=self.out_file,
                  lst=save_list[-1:],
                  method="append")

        for key in self.save_keys:
            setattr(self, key, getattr(self, key)[-5:])

    def ac_present(self,
                   old_list,
                   new_list):
        """
        Check if the previous AC step, whose properties you're updating,
        was actually saved.

        """

        # Our current calcs go
        # [AC on old surface (-3), step after AC (-2),
        # AC on new surface (-1)]
        # Meanwhile, the last thing saved, if saving every
        # step, was [AC on old surface (-1)]

        if len(old_list) == 0:
            present = False
            freq_gt_2 = False
            return present, freq_gt_2

        old_time = old_list[-1]["time"]
        new_time = new_list[-3]["time"]

        present = abs(old_time - new_time) < 1e-3

        # see if it saves every two frames. If so, it will save
        # the next calc. If not then it won't
        dt = abs(new_list[-1]["time"] - new_list[-2]["time"])
        old_time = old_list[-1]["time"]
        freq_gt_2 = len(old_list) * 2 * dt >= old_time

        return present, freq_gt_2

    def modify_hop(self, new_list):

        key = 'hopping_probability'
        new_list[-3][key] = copy.deepcopy(new_list[-2][key])
        new_list[-2][key] = []

        return new_list

    def modify_save(self):
        """
        Modify a saved csv file (e.g., with Zhu hopping parameters calculated in the next step)

        * this probably takes forever, and is more likely to happen as you increase the number of
        parallel trajectories

        """

        # old calculations, loaded from the csv
        old_list = csv_read(out_file=self.out_file)
        # new list of calculations, which only go back
        # 5 calcs
        new_list = self.create_save_list()

        # put the hopping probability info into the second
        # last calc, because that's where the hop really
        # happened
        new_list = self.modify_hop(new_list)

        # updated list of calculations with AC information
        # (e.g. hopped, hopping probability, in_trj=False,
        # etc.)
        save_list = copy.deepcopy(old_list)

        # Our current calcs go
        # [AC on old surface (-3), step after AC (-2),
        # AC on new surface (-1)]
        # Meanwhile, the last thing saved was
        # [AC on old surface (-1)] if every frame is
        # being saved.

        # check to see if [AC on old surface] was
        # actually saved - may not be if we're not
        # saving every frame
        ac_present, freq_gt_2 = self.ac_present(
            old_list=old_list,
            new_list=new_list)

        # update [AC on old surface] with the
        # fact that it's not in the trj and that
        # a hop occured

        if ac_present:
            # if it was already saved, then replace its
            # information
            save_list[-1] = new_list[-3]
        else:
            # if it wasn't saved, then add it to the save list
            save_list.append(new_list[-3])

        # append the calculation from step after AC
        save_list.append(new_list[-2])

        # save everything except [AC on new surface],
        # because that will be saved with the regular
        # save function in the next step anyway

        # but if we don't save frequently enough then we won't
        # save this calc. This would be a problem because
        # parsing identifies a hop geom as one whose surface
        # doesn't match the previous surface. So in that case
        # we add the calc.

        if not freq_gt_2:
            save_list.append(new_list[-1])

        csv_write(out_file=self.out_file,
                  lst=save_list,
                  method="new")

    def output_to_json(self):
        """
        Convert csv file to a json file.
        """

        props_list = csv_read(out_file=self.out_file)
        base_name = self.out_file.split(".csv")[0]
        json_name = f"{base_name}.json"
        with open(json_name, "w") as f:
            json.dump(props_list, f, indent=5, sort_keys=True)

    def log(self, msg):
        """
        Logs on the screen a message in the format of 'PREFIX:  msg' and also outputs to file

        Args:
            msg (str)
        """
        output = '{:>12}:  {}'.format("Zhu-Nakamura dynamics".upper(), msg)
        with open(self.log_file, "a+") as f:
            f.write(output)
            f.write("\n")
