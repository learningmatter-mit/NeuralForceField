import torch
import numpy as np

from graphbuilder.graphbuilder import Graph, GraphDataset


def species_geom_from_batches(graph_data, n_batches):
    '''Retrieve species, features and xyz from graphbuilder.GraphDataset
    '''

    species_dict = {}

    name_list = []
    r_list = []
    xyz_list = []

    for i in range(n_batch):
        batch = graph_data.batches[i]

        xyz_list += list(torch.split(batch.data['xyz'], batch.data['N']))
        r_list += list(torch.split(batch.data['r'], batch.data['N']))
        name_list += batch.data['name']

    for index, name in enumerate(name_list):
        species_dict[name] = species_dict.get(name, []) + [index]

    return species_dict, r_list, xyz_list
