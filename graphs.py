from sklearn.utils import shuffle
import numpy as np 
import torch 
from projects.graphbuilder.graphbuilder import Graph, GraphDataset

def load_graph_data(xyz_data, force_data, energy_data, batch_size, cutoff, au_flag, subtract_mean_flag, smiles_data=None):

    # shuffle data 
    xyz_data, force_data, energy_data = shuffle(xyz_data, force_data, energy_data)

    if subtract_mean_flag == True:
        energy_data = np.array(energy_data) - np.array(energy_data).mean()

    Fr = 4 # let node features be 
    Fe = 1
    dynamic_adj_mat = True
    graph_data = GraphDataset(dynamic_adj_mat=dynamic_adj_mat)

    # AU to kcal/mol A 
    if au_flag == True:
        force_conversion = 627.509 / 0.529177
        energy_conversion = 627.509
    else:
        force_conversion = 1.0
        energy_conversion = 1.0

    for index in range(len(energy_data)):

        xyz = np.array(xyz_data[index])
        force = np.array(force_data[index]) * np.float(force_conversion) 
        energy = np.array(energy_data[index]) * np.float(energy_conversion)
        species = smiles_data[index]
        node = xyz[:, 0].reshape(-1, 1)
        graph = Graph(N=node.shape[0], dynamic_adj_mat=dynamic_adj_mat, name=species)
        node_force = np.hstack((node, force)) # node concatenate with force 
        graph.SetNodeLabels(r=torch.Tensor(node_force))
        graph.SetXYZ(xyz=torch.Tensor(xyz[:, 1:4]))
        graph.UpdateConnectivity(cutoff=cutoff)
        graph.SetEdgeLabels()
        graph.LabelEdgesWithDistances()
        graph.SetGraphLabel(torch.Tensor([energy]))
        graph_data.AddGraph(graph)

    graph_data.CreateBatches(batch_size=batch_size)

    return graph_data

