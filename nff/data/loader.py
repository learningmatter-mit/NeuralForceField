import numpy as np 

from sklearn.utils import shuffle as skshuffle
import torch 

from nff.data import Graph, GraphDataset
import nff.utils.constants as const


class GraphLoader:
    """Dataloader to deal with NFF calculations. Can be expanded to retrieve calculations
         from the cluster later.

    Attributes:
        dataset (Dataset): dataset containing the information from htvs.
        batch_size (int)
        cutoff (float): if the distance between atoms is larger than
            cutoff, they are considered disconnected.
        device (int or 'cpu')
        shuffle (bool): if True, shuffle the dataset before creating the batches.
        dynamic_adj_mat (bool): if True, the Graphs are created with
            dynamic adjacency matrices.
        graph_dataset (GraphDataset): dataset from graphbuilder.
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
        """Constructor for GraphLoader class. Creates and interfaces a
            GraphDataset class from graphbuilder

        Args:
            dataset (Dataset): dataset containing the information from htvs.
            batch_size (int)
            cutoff (float): if the distance between atoms is larger than
                cutoff, they are considered disconnected.
            device (int or 'cpu')
            shuffle (bool): if True, shuffle the dataset before creating the batches.
            dynamic_adj_mat (bool): if True, the Graphs are created with
                dynamic adjacency matrices.
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.cutoff = cutoff
        self.dynamic_adj_mat = dynamic_adj_mat
        self.device = device

        if shuffle:
            self.dataset.shuffle()

        self.graph_dataset = self._init_graph_dataset(shuffle)

    def to(self, device):
        """Sends GraphLoader to the desired device"""
        self.device = device

    def __len__(self):
        return len(self.graph_dataset.batches)

    def __getitem__(self, idx):
        """Get batch `idx` from the GraphDataset
        
        Args:
            index (int): index of the batch in GraphDataset
        
        Returns:
            xyz (torch.Tensor)
            a (torch.Tensor)
            bond_adj (torch.Tensor or None)
            bond_len (torch.Tensor or None)
            r (torch.Tensor) 
            f (torch.Tensor)
            u (torch.Tensor)
            N (torch.Tensor)
            pbc (torch.Tensor)
        """

        neighbor_list = self.graph_dataset.batches[idx].data['neighbor_list'][:,1:].to(self.device)
        r = self.graph_dataset.batches[idx].data['r'][:, [0]].to(self.device)
        f = self.graph_dataset.batches[idx].data['r'][:, 1:4].to(self.device)
        u = self.graph_dataset.batches[idx].data['y'].to(self.device)
        N = self.graph_dataset.batches[idx].data['N']
        xyz = self.graph_dataset.batches[idx].data['xyz'].to(self.device)
        pbc = self.graph_dataset.batches[idx].data['pbc'].to(self.device)

        bond_adj = self.graph_dataset.batches[idx].data.get('bond_a', None)
        if bond_adj is not None:
            bond_adj.to(self.device)

        bond_len = self.graph_dataset.batches[idx].data.get('bond_len', None)
        if bond_len is not None:
            bond_len.to(self.device)

        return xyz, neighbor_list, bond_adj, bond_len, r, f, u, N, pbc

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
    
    def _init_graph_dataset(self, shuffle=True):
        """Uses graphbuilder to batch the dataset and create the dataloader.
    
        Args:
    
        Returns:
            graph_data (GraphDataset) 
        """
    
        if shuffle:
            self.shuffle()
        #energy_mean = np.mean(self.dataset.energy)
    
        graph_data = GraphDataset(dynamic=self.dynamic_adj_mat)
    
        for index in range(len(self.dataset)):
            nxyz = self.dataset.nxyz[index]
            force = self.dataset.force[index]
            energy = self.dataset.energy[index]
            smiles = self.dataset.smiles[index]
            pbc = self.dataset.pbc[index]
    
            number = nxyz[:, 0].reshape(-1, 1)

            graph = Graph(
                N=number.shape[0],
                dynamic=self.dynamic_adj_mat,
                pbc=pbc,
                directed=False,
                graphname=smiles
            )

            nforce = np.hstack((number, force))
            graph.SetNodeLabels(r=torch.Tensor(nforce))
            graph.SetXYZ(xyz=torch.Tensor(nxyz[:, 1:4]))
            graph.UpdateConnectivity(cutoff=self.cutoff)

            graph.LabelEdgesWithDistances()
            graph.SetGraphLabel(torch.Tensor([energy]))

            graph_data.AddGraph(graph)
    
        graph_data.CreateBatches(batch_size=self.batch_size, show_output=False)
    
        return graph_data

    def shuffle(self):
        self.dataset.shuffle()

    def shuffle_and_rebatch(self):
        self.shuffle()
        self.graph_dataset = self._init_graph_dataset()
