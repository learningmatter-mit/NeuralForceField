import torch 
import numpy as np 

from sklearn.utils import shuffle as skshuffle

from graphbuilder.graphbuilder import Graph, GraphDataset

import nff.utils.constants as const

class Loader:
    """Dataloader to deal with NFF calculations. Can be expanded to retrieve calculations
         from the cluster later.

    Attributes:
        nxyz (array): (N, 4) array with atomic number and xyz coordinates
            for each of the N atoms
        energy (array): (N, ) array with energies
        force (array): (N, 3) array with forces
        smiles (array): (N, ) array with SMILES strings
    """


    def __init__(
        self,
        dataset,
        batch_size,
        cutoff,
        device,
        shuffle=True,
        dynamic_adj_mat=True
    ):
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

        self.dataset = dataset
        self.batch_size = batch_size
        self.cutoff = cutoff
        self.dynamic_adj_mat = dynamic_adj_mat
        self.device = device

        if shuffle:
            self.dataset.shuffle()

        self.graph_dataset = self._init_graph_dataset()

    def __len__(self):
        return len(self.graph_dataset.batches)

    def __getitem__(self, idx):
        """Get batch `idx` from the GraphDataset
        
        Args:
            index (int): index of the batch in GraphDataset
        
        Returns:
            TYPE: Description
        """

        a = data.batches[idx].data["a"].to(self.device)
        r = data.batches[idx].data["r"][:, [0]].to(self.device)
        f = data.batches[idx].data["r"][:, 1:4].to(self.device)
        u = data.batches[idx].data["y"].to(self.device)
        N = data.batches[idx].data["N"]
        xyz = data.batches[index].data["xyz"].to(self.device)

        try: # try to get bond adjacency matrix 
            bond_adj = data.batches[index].data["bond_a"].to(self.device)
            bond_len = data.batches[index].data["bond_len"].to(self.device)
            return xyz, a, bond_adj, bond_len, r, f, u, N
        except:
            return xyz, a, r, f, u, N

    def __iter__(self):
        self.iter_n = 0
        return self

    def __next__(self):
        if self.iter_n < len(self):
            batch = self[self.iter_n]
            self.iter_n += 1
            return batch
        else:
            raise StopIteration
    
    def _init_graph_dataset(self):
        """Uses graphbuilder to batch the dataset and create the dataloader.
    
        Args:
    
        Returns:
            graph_data (GraphDataset) 
        """
    
        self.shuffle()
        energy_mean = np.mean(self.dataset.energy)
    
        graph_data = GraphDataset(dynamic_adj_mat=dynamic_adj_mat)
    
        for index in range(len(energy_data)):
            nxyz = self.dataset.nxyz[index]
            force = self.dataset.force[index]
            energy = self.dataset.energy[index]
            species = self.dataset.smiles[index]
    
            number = nxyz[:, 0].reshape(-1, 1)

            graph = Graph(N=number.shape[0],
                          dynamic_adj_mat=self.dynamic_adj_mat,
                          name=species)
    
            nforce = np.hstack((number, force))
            graph.SetNodeLabels(r=torch.Tensor(nforce))
            graph.SetXYZ(xyz=torch.Tensor(nxyz[:, 1:4]))
            graph.UpdateConnectivity(cutoff=self.cutoff)
            graph.SetEdgeLabels()
            graph.LabelEdgesWithDistances()
            graph.SetGraphLabel(torch.Tensor([energy]))
    
            graph_data.AddGraph(graph)
    
        graph_data.CreateBatches(batch_size=self.batch_size, verbose=False)
        graph_data.set_label_mean(energy_mean)
    
        return graph_data

    def shuffle(self):
        self.dataset.shuffle()
