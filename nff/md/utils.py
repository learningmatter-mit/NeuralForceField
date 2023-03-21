
import csv
import copy

from ase import units
from ase.md import MDLogger
from ase.utils import IOContext
import weakref
from ase.parallel import world

import numpy as np

import nff.utils.constants as const


class NeuralMDLogger(MDLogger):
    def __init__(self,
                 dyn, 
                 atoms, 
                 logfile, 
                 header=True, 
                 stress=False, 
                 peratom=False, 
                 mode="a",
                 verbose=True,
                 **kwargs):
        
        if hasattr(dyn, "get_time"):
            self.dyn = weakref.proxy(dyn)
        else:
            self.dyn = None
        self.atoms = atoms
        global_natoms = atoms.get_global_number_of_atoms()
        self.logfile = self.openfile(logfile, comm=world, mode=mode)
        self.stress = stress
        self.peratom = peratom

        if self.dyn is not None:
            self.hdr = "%-10s " % ("Time[ps]",)
            self.fmt = "%-10.4f "
        else:
            self.hdr = ""
            self.fmt = ""
        if self.peratom:
            self.hdr += "%12s %12s %12s  %6s" % ("Etot/N[eV]", "Epot/N[eV]",
                                                 "Ekin/N[eV]", "T[K]")
            self.fmt += "%12.4f %12.4f %12.4f  %6.1f"
        else:
            self.hdr += "%12s %12s %12s  %6s" % ("Etot[eV]", "Epot[eV]",
                                                 "Ekin[eV]", "T[K]")
            # Choose a sensible number of decimals
            if global_natoms <= 100:
                digits = 4
            elif global_natoms <= 1000:
                digits = 3
            elif global_natoms <= 10000:
                digits = 2
            else:
                digits = 1
            self.fmt += 3 * ("%%12.%df " % (digits,)) + " %6.1f"
        if self.stress:
            self.hdr += ('      ---------------------- stress [GPa] '
                         '------------------------')
            self.fmt += 6 * " %10.3f"
            self.hdr += "   %10s" % ("hydP[GPa]")
            self.fmt += "   %10.7f"
        
        self.fmt += "\n"
        if header:
            self.logfile.write(self.hdr + "\n")
        
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
            dat   += tuple(self.atoms.get_stress() / units.GPa)
            P_int  = -self.atoms.get_stress(include_ideal_gas=True, voigt=False)
            P_hyd  = np.trace(P_int)/len(P_int)
            dat   += (P_hyd / units.GPa,)
            
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()

        if self.verbose:
            print(self.fmt[:-1] % dat)


class NeuralFFLogger(MDLogger):
    def __init__(self,
                 dyn,
                 atoms,
                 logfile,
                 mode="a",
                 verbose=True,
                 **kwargs):
        # super().__init__(*args, **kwargs)

        self.atoms = atoms
        self.logfile = logfile
        self.verbose = verbose
        self.natoms = len(self.atoms)

    def __call__(self):
        if "embedding" in self.atoms.calc.properties:
            self.atoms.calc.log_embedding(self.atoms.calc.jobdir, self.logfile, self.atoms.get_embedding())

            
class BiasedNeuralMDLogger(IOContext):
    """Additional Class for logging biased molecular dynamics simulations.

    Parameters:
    dyn:           The dynamics.  Only a weak reference is kept.

    atoms:         The atoms.
    
    header:        Whether to print the header into the logfile.

    logfile:       File name or open file, "-" meaning standard output.

    mode="a":      How the file is opened if logfile is a filename.
    """
    def __init__(self,
                 dyn, 
                 atoms, 
                 logfile, 
                 header=True, 
                 peratom=False,
                 verbose=False,
                 mode="a"):

        if hasattr(dyn, "get_time"):
            self.dyn = weakref.proxy(dyn)
        else:
            self.dyn = None
            
        self.atoms   = atoms
        self.num_cv  = atoms.calc.num_cv
        self.n_const = atoms.calc.num_const
        
        self.logfile = self.openfile(logfile, 
                                     comm=world, 
                                     mode=mode)
        
        if self.dyn is not None:
            self.hdr = "%-10s " % ("Time[ps]",)
            self.fmt = "%-10.4f "
        else:
            raise ValueError("A dynamics object has to be attached to the logger!")
            
        self.hdr += "%12s %12s %12s " % ("U0+bias[eV]", 
                                   "U0[eV]",
                                  "AbsGradPot")
        self.fmt += "%12.5f %12.5f %12.4f "
            
        for i in range(self.num_cv):
            self.hdr += "%12s %12s %12s %12s %12s " % ("CV", "Lambda", "inv_m_cv",
                                                "AbsGradCV", "GradCV_GradU")
            self.fmt += "%12.4f %12.4f %12.4f %12.4f %12.4f "
            
        for i in range(self.n_const):
            self.hdr += "%12s " % ("Const")
            self.fmt += "%12.5f "
            
        self.fmt += "\n"
        if header:
            self.logfile.write(self.hdr + "\n")
        
        
        self.verbose = verbose
        if verbose:
            print(self.hdr)

    
    def __del__(self):
        self.close()
        
    def __call__(self):
        epot = self.atoms.get_potential_energy()
        epot_nobias = self.atoms.calc.get_property("energy_unbiased")
        absGradPot = self.atoms.calc.get_property("grad_length")
            
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000*units.fs)
            dat = (t,)
        else:
            dat = ()
        dat += (epot, epot_nobias, absGradPot)
        
        cv_vals    = self.atoms.calc.get_property("cv_vals")
        ext_pos    = self.atoms.calc.get_property("ext_pos")
        cv_invmass = self.atoms.calc.get_property("cv_invmass")
        absGradCV  = self.atoms.calc.get_property("cv_grad_lengths")
        CVdotPot   = self.atoms.calc.get_property("cv_dot_PES")
        if self.n_const > 0:
            consts = self.atoms.calc.get_property('const_vals')
        
        for i in range(self.num_cv):
            dat += (cv_vals[i], ext_pos[i], cv_invmass[i], absGradCV[i], CVdotPot[i])
            
        for i in range(self.n_const):
            dat += (consts[i])

        self.logfile.write(self.fmt % dat)
        self.logfile.flush()

        if self.verbose:
            print(self.fmt[-1] % dat)


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
