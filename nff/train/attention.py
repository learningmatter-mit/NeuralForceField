import os
import json
import datetime
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

import torch.optim as optim

from nff.nn.models import *
from nff.utils.scatter import *
from nff.md.MD import * 

class ModelAttention():

    """Summary
    
    Attributes:
        criterion (MSEloss): Description
        data (graph): Description
        device (TYPE): Description
        dir_loc (TYPE): Description
        energy_mae (float): Description
        f_predict (list): Description
        f_true (list): Description
        force_mae (float): Description
        graph_batching (Boolean): If True, use graph batch input
        job_name (str): name of the job or experimeent
        mae (TYPE): Description
        model (TYPE): Description
        model_path (TYPE): Description
        N_batch (int): Description
        N_test (int): Description
        N_train (int): Description
        optimizer (TYPE): Description
        par (dict): a dictionary file for hyperparameters
        root (str): the root path for the saving training results 
        scheduler (TYPE): Description
        train_f_log (list): Description
        train_u_log (list): Description
        u_predict (list): Description
        u_true (list): Description
    """

    def __init__(self,par, graph_data, device, job_name, graph_batching=False, root="./"):
        """Summary
        
        Args:
            par (TYPE): Description
            graph_data (TYPE): Description
            device (TYPE): Description
            job_name (TYPE): Description
            graph_batching (bool, optional): Description
            root (str, optional): Description
        """
        self.device = device
        self.par = par 
        self.data = graph_data
        self.job_name = job_name
        self.root = root 
        self.initialize_log()
        self.initialize_data()
        self.initialize_model()
        self.initialize_optim()
        self.graph_batching = graph_batching
        
    def initialize_model(self):

        with open(self.dir_loc + "/par.json", "w") as write_file:
            json.dump(self.par, write_file, indent=4)
        
        self.model = Net_att(n_atom_basis = self.par["n_atom_basis"],
                            n_filters = self.par["n_filters"],
                            n_gaussians= self.par["n_gaussians"], 
                            cutoff_soft= self.par["cutoff"], 
                            trainable_gauss = self.par["trainable_gauss"],
                            T = self.par["T"],
                            device= self.device).to(self.device)
    
    def initialize_optim(self):
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=self.par["optim"])
        
        if self.par["scheduler"]:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                  'min', 
                                                                  min_lr=1.5e-7, 
                                                                  verbose=True, factor = 0.5, patience= 50,
                                                                  threshold= 5e-5)
        self.criterion = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
        
    def initialize_log(self):
        
        self.train_u_log = []
        self.train_f_log = []

        # create directoires if not exists 
        if not os.path.exists(self.root):
            os.makedirs(self.root)
            
        # obtain a time stamp 
        currentDT = datetime.datetime.now()
        date = str(currentDT).split()[0].split("-")[1:] 
        self.dir_loc = self.root + self.job_name + "_" + "".join(date)
        
        if not os.path.exists(self.dir_loc):
            os.makedirs(self.dir_loc)
        
    def initialize_data(self):
        
        self.N_batch = len(self.data.batches)
        self.N_train = int(self.par["train_percentage"] * self.N_batch)
        self.N_test = self.N_batch - self.N_train - 1 # ignore the last batch 
        
    def parse_batch(self, index, data=None):
        """Summary
        
        Args:
            index (int): index of the batch in GraphDataset
        
        Returns:
            TYPE: Description
        """
        if data == None:
            data = self.data

        a = data.batches[index].data["a"].to(self.device)

        r = data.batches[index].data["r"][:, [0]].to(self.device)

        f = data.batches[index].data["r"][:, 1:4].to(self.device) #* (627.509 / 0.529177) # convert to kcal/mol A

        u = data.batches[index].data["y"].to(self.device) #* 627.509 # kcal/mol 
        
        N = data.batches[index].data["N"]
        
        xyz = data.batches[index].data["xyz"].to(self.device)
        
        return xyz, a, r, f, u, N
    
    def train(self, N_epoch):
        """Summary
        
        Args:
            N_epoch (TYPE): Description
        """

        self.start_time = time.time()

        for epoch in range(N_epoch):

            # check if max epoches are reached 
            if len(self.train_f_log) >= self.par["max_epoch"]:
                print("max epoches reached")
                break 

            train_u_mae = 0.0
            train_force_mae = 0.0
            
            for i in range(self.N_train):

                xyz, a, r, f, u, N = self.parse_batch(i)
                xyz.requires_grad = True

                # check if the input has graphs of various sizes 
                if len(set(N)) == 1:
                    graph_size_is_same = True
                else:
                    graph_size_is_same = False
                
                # Compute energies 
                if self.graph_batching:
                    U = self.model(r=r, xyz=xyz, a=a, N=N)
                else:
                    assert graph_size_is_same # make sure all the graphs needs to have the same size
                    U = self.model(r=r.reshape(-1, N[0]), xyz=xyz.reshape(-1, N[0], 3))
                    
                f_pred = -compute_grad(inputs=xyz, output=U)

                # comput loss
                loss_force = self.criterion(f_pred, f)
                loss_u = self.criterion(U, u)
                loss = loss_force + self.par["rho"] * loss_u

                # update parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # compute MAE
                train_u_mae += self.mae(U, u) # compute MAE
                train_force_mae += self.mae(f_pred, f)

            # averaging MAE
            train_u = train_u_mae.data[0]/self.N_train
            train_force = train_force_mae.data[0]/self.N_train
            
            self.train_u_log.append(train_u.item())
            self.train_f_log.append(train_force.item())
            
            # scheduler
            if self.par["scheduler"] == True:
                self.scheduler.step(train_force)
            else:
                pass

            # print loss
            print("epoch %d  U train: %.3f  force train %.3f" % (epoch, train_u, train_force))

            # check convergence 
            #if self.check_convergence():
            #    print("training converged")
            #    break
            #else:
            #    pass

        self.time_elapsed = time.time() - self.start_time

        #self.save_model()
        #self.save_train_log()
