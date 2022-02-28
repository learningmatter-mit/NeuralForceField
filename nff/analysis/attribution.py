import torch
from ase.io import Trajectory

from nff.io.ase import EnsembleNFF
from nff.io.ase import AtomsBatch
from nff.utils.scatter import compute_grad
from nff.utils.cuda import batch_to


class Attribution:

    def __init__(self, ensemble: EnsembleNFF):
        self.ensemble = ensemble
        
    @property
    def device(self):
        return self.ensemble.device

    def __call__(self, atoms: AtomsBatch):
        atoms.calc = self.ensemble
        atoms.update_nbr_list()
        try:
            atoms.update_mol_nbrs_list()
        except:
            pass
        batch = batch_to(atoms.get_batch(), self.device)
        batch['nxyz'].requires_grad=True
        xyz=batch['nxyz'][:,1:]

        # why this??? xyz is contained in the batch
        results = [
                        m(batch,xyz=xyz)
                    for m in self.ensemble.models
                ]

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

        return Z1.detach().cpu().numpy()

    def calc_attribution_trajectory(self, 
        traj: Trajectory, 
        directed: bool,
        cutoff: float,
        requires_large_offsets: bool,
        skip:int = 0, 
        step:int = 1
    )->list:
        attributions = []
        for i in range(skip,len(traj),step):
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
            attributions.append(self(atoms))
        return attributions