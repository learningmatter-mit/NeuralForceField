import json
import numpy as np
import torch

class Data():

    """Data Object
    
    Attributes:
        energy_test (TYPE): Description
        energy_train (TYPE): Description
        force_test (TYPE): Description
        force_train (TYPE): Description
        node_test (TYPE): Description
        node_train (TYPE): Description
        xyz_test (TYPE): Description
        xyz_train (TYPE): Description
    """

    def __init__(self, par):
        """Summary
        
        Args:
            par (dict): dictionary that contains all the hypar
        """
        
        N_data = par["N_data"]
        train_percent = par["train_percent"]

        batch_size = par["batch_size"]
        data = np.load(par["data"])


        xyz_data = data.f.R
        force_data = data.f.F
        energy_data = data.f.E
        z_data = data.f.z
        n_atom = z_data.shape[0]

        if N_data < data.f.R.shape[0]:
            pass
        else:
            N_data = data.f.R.shape[0]

        N_train = int(N_data * train_percent)
        N_test = N_data - N_train 

        xyz_data = xyz_data[:N_train + N_test]
        force_data = force_data[:N_train + N_test]
        energy_data = energy_data[:N_train + N_test]

        #Shuffle data 
        indices = np.arange(N_data)
        indices = np.random.shuffle(indices)

        xyz_data, force_data, energy_data = xyz_data[indices], force_data[indices], energy_data[indices]

        xyz_train = xyz_data[:N_train]
        force_train = force_data[:N_train]
        energy_train = energy_data[:N_train]

        xyz_test = xyz_data[-N_test:]
        force_test = force_data[-N_test:]
        energy_test = energy_data[-N_test:]

        N_batch_train = (N_train // batch_size) * batch_size
        N_batch_test = (N_test // batch_size) * batch_size

        #prepare data 
        self.node_train = torch.Tensor(data.f.z).reshape(1, n_atom, 1).repeat(batch_size, 1, 1)
        self.xyz_train = torch.Tensor(xyz_train[0][:N_batch_train]).reshape(-1, batch_size, n_atom, 3)
        self.force_train = torch.Tensor(force_train[0][:N_batch_train]).reshape(-1, batch_size, n_atom, 3)
        self.energy_train = torch.Tensor(energy_train[0][:N_batch_train]).reshape(-1, batch_size, 1)

        self.node_test = torch.Tensor(data.f.z).reshape(1, n_atom,1).repeat(batch_size, 1, 1)
        self.xyz_test = torch.Tensor(xyz_test[0][:N_batch_test]).reshape(-1, batch_size, n_atom, 3)
        self.force_test = torch.Tensor(force_test[0][:N_batch_test]).reshape(-1, batch_size, n_atom, 3)
        self.energy_test = torch.Tensor(energy_test[0][:N_batch_test]).reshape(-1, batch_size, 1)