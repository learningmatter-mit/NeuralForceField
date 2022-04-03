import torch
from ase.io import Trajectory
from ase import Atoms
import numpy as np

from nff.io.ase import EnsembleNFF
from nff.io.ase import AtomsBatch
from nff.utils.scatter import compute_grad
from nff.utils.cuda import batch_to

from tqdm import tqdm


class Attribution:

    def __init__(self, ensemble: EnsembleNFF):
        self.ensemble = ensemble
        
    @property
    def device(self):
        return self.ensemble.device

    def __call__(self, atoms: AtomsBatch):
        atoms.calc = self.ensemble
        atoms.update_nbr_list()
        batch = batch_to(atoms.get_batch(), self.device)
        batch['nxyz'].requires_grad=True
        xyz=batch['nxyz'][:,1:]

        results = [
                        m(batch,xyz=xyz)
                    for m in self.ensemble.models
                ]

        # we also return energies just to get it running in chemiscope
        # have to further think about it 
        energy = torch.cat([r['energy']
                    for r in results
                ]).mean()

        energy_std = torch.cat([r['energy']
                    for r in results
                ]).std()

        forces = torch.stack([
                    r['energy_grad']
                    for r in results
                ], dim=-1)

        
        # calculate gradient of the standard deviation of the forces per atom 
        # wrt the coordinates, first var is to get the variance per atom coordination (x, y, z)
        # second mean is to take then the average of this variance per atom cooridnation 
        # resulting in one scalar per atom
        # computing the grad wrt the coordinates results in a 3dim tensor
        Z3=compute_grad(xyz,-forces.var(-1).mean(-1))

        # average over the 3 dimensions and get one number per atom back
        # which is the actual attribution
        Z1=abs(Z3).mean(-1)

        return Z1.detach().cpu().numpy(), energy.detach().cpu().numpy(), energy_std.detach().cpu().numpy(), forces.detach().cpu().numpy().mean(-1), forces.detach().cpu().numpy().std(-1)

    def calc_attribution_trajectory(self, 
        traj: Trajectory, 
        directed: bool,
        cutoff: float,
        requires_large_offsets: bool,
        skip:int = 0, 
        step:int = 1,
        progress_bar:bool = True,
        to_chemiscope:bool = False,
    )->list:
        attributions = []
        atoms_list = []
        energies = []
        energy_stds = []
        grads = []
        grad_stds = []
        with tqdm(range(skip,len(traj),step),disable=True if progress_bar == False else False) as pbar:#, postfix={"fbest":"?",}) as pbar:
            
            #for i in range(skip,len(traj),step):
            for i in pbar:
                # create atoms batch object
                atoms = AtomsBatch(
                    positions=traj[i].get_positions(wrap=True),
                    cell=traj[i].cell,
                    pbc=traj[i].pbc,
                    numbers=traj[i].numbers,props={},
                    directed=directed,
                    cutoff=cutoff,
                    requires_large_offsets=requires_large_offsets,
                    device=self.device,
                )
                attribution, energy, energy_std, grad, grad_std = self(atoms)
                if to_chemiscope: 
                    atoms_list.append(Atoms(
                        positions = atoms.positions,
                        numbers = atoms.numbers,
                        cell = atoms.cell,
                        pbc = atoms.pbc
                    ))
                    attributions.append(attribution)
                    energies.append(energy)
                    energy_stds.append(energy_std)
                    grads.append(grad)
                    grad_stds.append(grad_std)

        if to_chemiscope:
            properties = {
                "attribution": {
                    "target": "atom",
                    "values": np.concatenate(attributions),
                },
                "energy": {
                    "target": "structure",
                    "values": np.array(energies),
                    "units": "eV",
                },
                "energy_std": {
                    "target": "structure",
                    "values": np.array(energy_stds),
                    "units": "eV",
                },
                "energy_grad": {
                    "target": "atom",
                    "values": np.concatenate(grads),
                    "units": "eV/A",
                },
                "energy_grad_std": {
                    "target": "atom",
                    "values": np.concatenate(grad_stds),
                    "units": "eV/A",
                },
            }
            return atoms_list, properties
        else:
            return attributions
