import torch 
import numpy as np 

from sklearn.utils import shuffle as skshuffle

from graphbuilder import Graph, GraphDataset


HARTREE_TO_KCAL_MOL = 627.509
BOHR_RADIUS = 0.529177


class Dataset:
    """Dataset to deal with NFF calculations. Can be expanded to retrieve calculations
         from the cluster later.

    Attributes:
        nxyz (array): (N, 4) array with atomic number and xyz coordinates
            for each of the N atoms
        energy (array): (N, ) array with energies
        force (array): (N, 3) array with forces
        smiles (array): (N, ) array with SMILES strings
    """


    array_type = np.array

    def __init__(self,
                 nxyz,
                 energy,
                 force,
                 smiles,
                 atomic_units=False):
        """Constructor for Dataset class.

        Args:
            nxyz (array): (N, 4) array with atomic number and xyz coordinates
                for each of the N atoms
            energy (array): (N, ) array with energies
            force (array): (N, 3) array with forces
            smiles (array): (N, ) array with SMILES strings
            atomic_units (bool): if True, input values are given in atomic units.
                They will be converted to kcal/mol.
        """

        self.nxyz = array_type(nxyz)
        self.energy = array_type(energy)
        self.force = array_type(force)
        self.smiles = array_type(smiles)

        if atomic_units:
            units_to_kcal_mol()

    def units_to_kcal_mol(self):
        """Converts forces and energies from atomic units to kcal/mol."""
    
        self.force = self.force * HARTREE_TO_KCAL_MOL / BOHR_RADIUS
        self.energy = self.energy * HARTREE_TO_KCAL_MOL 
    
    def to_graph_dataset(self, batch_size, cutoff, atomic_units=False, dynamic_adj_mat=True):
        """Loads the dataset under consideration.
    
        Args:
            dataset (dict of lists): dicionary containing the xyz, forces, energy, smiles
            batch_size (int): size of the batch
            atomic_units (bool): if True, convert the input units from atomic units to kcal/mol
            dynamic_adj_mat (bool): if True, WUJIE
    
        Returns:
        """
    
        self.shuffle()
        energy_mean = self.energy.mean()
    
        graph_data = GraphDataset(dynamic_adj_mat=dynamic_adj_mat)
    
        for index in range(len(energy_data)):
            nxyz = self.nxyz[index]
            force = self.force[index]
            energy = self.energy[index]
            species = self.smiles[index]
    
            number = self.nxyz[:, 0].reshape(-1, 1)
            graph = Graph(N=number.shape[0],
                          dynamic_adj_mat=dynamic_adj_mat,
                          name=species)
    
            nforce = np.hstack((number, self.force))
            graph.SetNodeLabels(r=torch.Tensor(nforce))
            graph.SetXYZ(xyz=torch.Tensor(self.nxyz[:, 1:4]))
            graph.UpdateConnectivity(cutoff=cutoff)
            graph.SetEdgeLabels()
            graph.LabelEdgesWithDistances()
            graph.SetGraphLabel(torch.Tensor([energy]))
    
            # WUJIE: adj_dict
    
            graph_data.AddGraph(graph)
    
        graph_data.CreateBatches(batch_size=batch_size, verbose=False)
        graph_data.set_label_mean(energy_mean)
    
        return graph_data
    
    def shuffle(self):
        self.nxyz, self.forces, self.energies, self.smiles = skshuffle(
            self.nxyz, self.forces, self.energies, self.smiles
        )
        return shuffled_dataset
