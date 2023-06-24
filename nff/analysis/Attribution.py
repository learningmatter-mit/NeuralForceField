import torch
from ase.io import Trajectory,write
from ase import Atoms
import numpy as np

from nff.io.ase import EnsembleNFF
from nff.io.ase import AtomsBatch
from nff.utils.scatter import compute_grad
from nff.utils.cuda import batch_to
from typing import Union

from tqdm import tqdm



def get_molecules(atom: AtomsBatch,
                  bond_length: dict = None,
                  mode: str ='bond',
                  **kwargs) -> list[np.array]:
    '''
    find molecules in periodic or non-periodic system. bond mode finds molecules within bond length. 
    Must pass bond_length dict: e.g bond_length=dict()
    bond_length['8-1']=1.4
    bond_length['8-14']=1.9
    bond_length['1-8']=bond_length['8-1']
    bond_length['14-8']=bond_length['8-14']....
    bonds for Si-O and O-H defined
    
    inputs:
    atoms: Atomsbatch object from NFF
    bond_length: dict for bond lengths for your elements
    mode: bond: chooses atoms in molecules based on bonds; cutoff: chooses atoms in clusters within sphere of cutoff; 
    give extra cutoff = 6 e.g input
    
    output:
    list of array of atom indices in molecules. e.g: if there is a H2O molecule, you will get a list with the atom indices
    
    '''
    types=list(set(atom.numbers))
    xyz=atom.positions
    #A=np.lexsort((xyz[:,2],xyz[:,1],xyz[:,0]))
    dis_mat = xyz[None, :, :] - xyz[:, None, :]
    if any(atom.pbc):
        cell_dim = np.diag(np.array(atom.get_cell()))
        shift = np.round(np.divide(dis_mat,cell_dim))
        offsets = -shift
        dis_mat=dis_mat+offsets*cell_dim
    dis_sq = torch.tensor(dis_mat).pow(2).sum(-1).numpy()
    dis_sq=dis_sq**0.5
    clusters=np.array([0 for i in range(xyz.shape[0])])
    for i in range(xyz.shape[0]):
        mm=max(clusters)
        ty=atom.numbers[i]
        oxy_neighbors=[]
        if mode=='bond':
            for t in types:
                if bond_length.get('%s-%s'%(ty,t))!=None:
                    oxy_neighbors.extend(list(np.where(atom.numbers==t)[0][np.where(dis_sq[i,np.where(atom.numbers==t)[0]]<=bond_length['%s-%s'%(ty,t)])[0]]))
        elif mode=='cutoff':
            oxy_neighbors.extend(list(np.where(dis_sq[i]<=cutoff)[0])) # cutoff input extra argument
        oxy_neighbors=np.array(oxy_neighbors)
        if len(oxy_neighbors)==0:
            clusters[i]=mm+1
            continue
        if (clusters[oxy_neighbors]==0).all() and clusters[i]!=0:
            clusters[oxy_neighbors]=clusters[i]
        elif (clusters[oxy_neighbors]==0).all() and clusters[i]==0:
            clusters[oxy_neighbors]=mm+1
            clusters[i]=mm+1
        elif (clusters[oxy_neighbors]==0).all() == False and clusters[i]==0:
            clusters[i]=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])
            clusters[oxy_neighbors]=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])
        elif (clusters[oxy_neighbors]==0).all() == False and clusters[i]!=0:
            tmp=clusters[oxy_neighbors][clusters[oxy_neighbors]!=0][clusters[oxy_neighbors][clusters[oxy_neighbors]!=0]!=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])]
            clusters[i]=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])
            clusters[oxy_neighbors]=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])
            for tr in tmp:
                clusters[np.where(clusters==tr)[0]]=min(clusters[oxy_neighbors][clusters[oxy_neighbors]!=0])
            
    molecules=[]
    for i in range(1,max(clusters)+1):
        if np.size(np.where(clusters==i)[0])==0:
            continue
        molecules.append(np.where(clusters==i)[0])

    return molecules 
def reconstruct_atoms(atomsobject: AtomsBatch,
                      mol_idx: list[np.array],
                      centre:int =None):
    '''
    Function to shift atoms when we create non-periodic system from periodic.  
    inputs:
    atomsobject: Atomsbatch object from NFF
    mol_idx: list of array of atom indices in molecules or atoms you want to keep together when changing to non-periodic
    system
    centre: by default the atoms in a molecule or set of close atoms are shifted so as to get them close to the centre which 
    is by default the first atom index in the array. For reconstructing molecules this is fine. However, for attribution,
    we may have to shift a whole molecule to come closer to the atoms with high attribution. In that case, we manually assign 
    the atom index. 
    '''
    
    sys_xyz = torch.Tensor(atomsobject.get_positions(wrap=True))
    box_len = torch.Tensor(atomsobject.get_cell_lengths_and_angles()[:3])
    
    for idx in mol_idx:
        mol_xyz = sys_xyz[idx]
        if any(atomsobject.pbc):
            center = mol_xyz.shape[0]//2
            if centre!=None:
                center=centre # changes the central atom to atom in focus
            intra_dmat = (mol_xyz[None, :,...] - mol_xyz[:, None, ...])[center]
            if np.count_nonzero(atomsobject.cell.T-np.diag(np.diagonal(atomsobject.cell.T)))!=0:
                M,N=intra_dmat.shape[0],intra_dmat.shape[1]
                f=torch.linalg.solve(torch.Tensor(atomsobject.cell.T),(intra_dmat.view(-1,3).T)).T
                g=f-torch.floor(f+0.5)
                intra_dmat=torch.matmul(g,torch.Tensor(atomsobject.cell))
                intra_dmat=intra_dmat.view(M,3)
                offsets=-torch.floor(f+0.5).view(M,3)
                traj_unwrap = mol_xyz+torch.matmul(offsets,torch.Tensor(atomsobject.cell))
            else:
                sub = (intra_dmat > 0.5 * box_len).to(torch.float) * box_len
                add = (intra_dmat <= -0.5 * box_len).to(torch.float) * box_len
                shift=torch.round(torch.divide(intra_dmat,box_len))
                offsets=-shift
                traj_unwrap = mol_xyz+offsets*box_len
        else:
            traj_unwrap = mol_xyz
        #traj_unwrap=mol_xyz+add-sub
        sys_xyz[idx] = traj_unwrap

    new_pos = sys_xyz.numpy()

    return new_pos  


# -

class Attribution:

    def __init__(self, ensemble: EnsembleNFF, save_file: str = None):
        self.ensemble = ensemble
        self.save_file= save_file
    @property
    def device(self):
        return self.ensemble.device

    def __call__(self, atoms: AtomsBatch):
        atoms.calc = self.ensemble
        atoms.update_nbr_list()
        atoms.mol_nbrs,atoms.mol_idx = None,None
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

    def calc_attribution_file(self, 
        traj: Union[Trajectory,Atoms], 
        directed: bool = True,
        cutoff: float = 6,
        requires_large_offsets: bool = True,
        skip:int = 0, 
        step:int = 1,
        progress_bar:bool = True,
        to_chemiscope:bool = False,
        bond_length: dict = None,
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
                atoms.arrays['attribute'] = attribution
                if to_chemiscope: 
                    atoms_list.append(Atoms(
                        positions = atoms.positions,
                        numbers = atoms.numbers,
                        cell = atoms.cell,
                        pbc = atoms.pbc
                    ))
                if self.save_file is not None:
                    molecules=get_molecules(atoms,bond_length)
                    xyz=reconstruct_atoms(atoms,molecules)
                    atoms.positions=xyz
                    atoms.pbc=False
                    atoms_list.append(atoms)
                    write(self.save_file,atoms_list,write_results=True)
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
    def activelearning(self, 
        traj: Union[Trajectory,Atoms], 
        directed: bool = True,
        cutoff: float = 6,
        requires_large_offsets: bool = True,
        skip:int = 0, 
        step:int = 1,
        progress_bar:bool = True,
        bond_length: dict = None,
    ):
        atom_list=[]
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
                atoms.arrays['attribute'] = attribution
                molecules=get_molecules(atoms,bond_length)
                xyz=reconstruct_atoms(atoms,molecules)
                atoms.positions=xyz
                atoms.pbc=False
                uncertainatoms=np.where(attribution>=(attribution.mean()+2*attribution.std()))[0]
                moleculess=molecules.copy()
                row_lengths=[]
                for j,row in enumerate(moleculess):
                    moleculess[j]=list(row)
                    row_lengths.append(len(row))

                max_length = max(row_lengths)

                for row in moleculess:
                    while len(row) < max_length:
                        row.append(None)

                balanced_mols = np.array(moleculess)

                xyz=atoms.positions
                dis_mat = xyz[None, :, :] - xyz[:, None, :]
                if any(atoms.pbc):
                    cell_dim = np.diag(np.array(atoms.get_cell()))
                    shift = np.round(np.divide(dis_mat,cell_dim))
                    offsets = -shift
                    dis_mat=dis_mat+offsets*cell_dim
                dis_sq = torch.tensor(dis_mat).pow(2).sum(-1).numpy()
                dis_sq=dis_sq**0.5
                for a in uncertainatoms:
                    atomstocare=np.array([])
                    neighs=np.where(dis_sq[a]<=6)[0]
                    neighs=np.append(neighs,a)
                    for n in neighs:
                        atomstocare=np.append(atomstocare,molecules[np.where(balanced_mols==n)[0][0]])
                    atomstocare=np.array((list(set(atomstocare))))
                    atomstocare=np.int64(atomstocare)
                    atoms1=atoms[atomstocare]
                    index=np.where(atoms1.positions==atoms.positions[a])[0][0]
                    xyz=reconstruct_atoms(atoms1,[np.arange(0,len(atoms1))],centre=index)
                    atoms1.positions=xyz
                    is_repeated =False
                    for Atoms in atom_list:
                        if atoms1.__eq__(Atoms):
                            is_repeated = True
                            break
                    if not is_repeated:
                        atom_list.append(atoms1)
    
        clusterfile= 'cluster'+self.save_file
        write(clusterfile,atom_list,format='xyz')


