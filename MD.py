from torch.autograd import Variable
from projects.NeuralForceField.scatter import compute_grad
import torch

import ase
from ase.calculators.calculator import Calculator, all_changes
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase import Atoms

mass_dict = {6: 12.01, 8: 15.999, 1: 1.008, 3: 6.941}


def mol_state(r, xyz):
    mass = [mass_dict[item] for item in r]
    atom = "C" * r.shape[0] # intialize Atom()
    structure = Atoms(atom, positions=xyz, cell=[100.0, 100.0, 100.0], pbc=True)
    structure.set_atomic_numbers(r)
    structure.set_masses(mass)    
    return structure

def printenergy(atoms):
    """Function to print the potential, kinetic and total energy"""
    epot = atoms.get_potential_energy() / len(a)
    ekin = atoms.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

def write_traj(filename, frames):
    '''
        Write trajectory dataframes into .xyz format for VMD visualization
        to do: include multiple atom types 
        
        example:
            path = "../../sim/topotools_ethane/ethane-nvt_unwrap.xyz"
            traj2write = trajconv(n_mol, n_atom, box_len, path)
            write_traj(path, traj2write)
    '''    
    file = open(filename,'w')
    atom_no = frames.shape[1]
    for i, frame in enumerate(frames): 
        file.write( str(atom_no) + '\n')
        file.write('Atoms. Timestep: '+ str(i)+'\n')
        for atom in frame:
            if atom.shape[0] == 4:
                file.write(str(int(atom[0])) + " " + str(atom[1]) + " " + str(atom[2]) + " " + str(atom[3]) + "\n")
            elif atom.shape[0] == 3:
                file.write("1" + " " + str(atom[0]) + " " + str(atom[1]) + " " + str(atom[2]) + "\n")
            else:
                raise ValueError("wrong format")
    file.close()

class NeuralMD(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, device, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # number of atoms 
        n_atom = atoms.get_atomic_numbers().shape[0]
        
        # run model 
        node = atoms.get_atomic_numbers().reshape(1, -1, 1)
        xyz = atoms.get_positions().reshape(-1, n_atom, 3)

        node = Variable(torch.LongTensor(node).expand(2, n_atom, 1)).cuda(self.device)
        xyz = Variable(torch.Tensor(xyz).expand(2, n_atom, 3)).cuda(self.device)
        xyz.requires_grad = True

        # predict energy and force
        U = self.model(r=node, xyz=xyz)
        f_pred = -compute_grad(inputs=xyz, output=U)
        
        # change energy and force to numpy array 
        energy = U[0].detach().cpu().numpy() * 0.043
        forces = f_pred[0].detach().cpu().numpy() * 0.043
        
        self.results = {
            'energy': energy.reshape(-1),
            'forces': forces.reshape((len(atoms), 3))
        }