import os
import json
import datetime
import time
import pickle
import numpy as np

import torch.optim as optim

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from nff.nn.models import *
from nff.utils.scatter import *
from nff.md import * 
from nff.data.graphs import * 


class TrainWrapper:
    """A wrapper for training, validation, save and load models.
    
    Attributes:
        criterion (MSEloss): Description
        data (graph): Description
        device (TYPE): Description
        dir_loc (TYPE): Description
        energiesmae (float): Description
        predictedforces (list): Description
        targetforces (list): Description
        forcesmae (float): Description
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
        scheduler (Boolean): Description
        train_f_log (list): Description
        train_u_log (list): Description
        predictedenergies (list): Description
        targetenergies (list): Description
    """

    def __init__(self,par, device, job_name, graph_batching=True,
                 graph_data=None,  root="./", train_flag=False, shift=False):
        """Summary
        
        Args:
            par (TYPE): Description
            graph_data (TYPE): Description
            device (TYPE): Description
            job_name (TYPE): Description
            graph_batching (bool, optional): Description
            root (str, optional): Description
        """
        if graph_data == None and train_flag == True:
            raise ValueError("No graph data provided for training")
        if graph_data is not None and train_flag == False:
            raise ValueError("You import a graph dataset but dont want to train on the data, are you sure?")
        self.device = device
        self.job_name = job_name
        self.root = root 
        self.train_flag = train_flag
        self.par = par # needs to input parameters 
        self.check_parameters()

        if graph_data is not None:
            self.data = graph_data
            self.initialize_data()
            
        self.initialize_model()
        self.initialize_optim()
        
        if train_flag is False:
            print("need to load a pre-trained model")
        self.graph_batching = graph_batching
        
    def check_parameters(self):
        assert type(self.par["n_filters"]) == int, "Invalid filter dimension, it should be an integer"
        assert type(self.par["n_gaussians"]) == int, "Invalid number of gaussian basis, it should be an integer"
        assert type(self.par["optim"]) == float, "the learning rate is not an float"
        assert type(self.par["train_percentage"]) == float and self.par["optim"] < 1.0, "the training data percentage is invalid"
        assert type(self.par["T"]) == int, "number of convolutions have to an integer"
        assert type(self.par["batch_size"]) == int, "invalid batch size"
        assert type(self.par["cutoff"]) == float, "Invalid cutoff radius"
        assert type(self.par["max_epoch"]) == int, "max epoch should be an integer"
        assert type(self.par["trainable_gauss"]) == bool, "should be boolean value"
        assert type(self.par["rho"]) == float, "Rho should be float"
        assert type(self.par["eps"]) == float and self.par["eps"] <= 1.0, "Invalid convergence criterion"

    def initialize_model(self):

        if self.train_flag == True:

            print("setting up directories for saving training files")

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

            with open(self.dir_loc + "/par.json", "w") as write_file:
                json.dump(self.par, write_file, indent=4)
        
        bondpar = self.par.get("bondpar", 50.0)
        box_vec = self.par.get("box_vec", None)

        self.model = BondNet(n_atom_basis = self.par["n_atom_basis"],
                             n_filters = self.par["n_filters"],
                             n_gaussians= self.par["n_gaussians"], 
                             cutoff_soft= self.par["cutoff"], 
                             trainable_gauss = self.par["trainable_gauss"],
                             T=self.par["T"],
                             device=self.device,
                             bondpar=bondpar,
                             box_len=box_vec).to(self.device)        
    
    def initialize_optim(self):
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=self.par["optim"])
        
        if self.par["scheduler"]:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                                  'min', 
                                                                  min_lr=1.5e-7, 
                                                                  verbose=True, factor = 0.5, patience= 20,
                                                                  threshold=5e-5)
        self.criterion = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
         
    def plot_results(self):
        fig, ax = plt.subplots(1, 2, figsize=(13,6))

        ax[0].set_title("Energies (validation)")
        ax[1].set_title("Forces (validation)")
    
        ax[1].scatter(self.targetforces,
                   self.predictedforces,
                   label="force MAE: " + str(self.forcesmae) + " kcal/mol A",
                   alpha=0.3,
                   s=6)

        ax[1].set_xlabel("test")
        ax[1].set_ylabel("prediction")
        ax[1].legend()

        ax[0].scatter(self.targetenergies, self.predictedenergies, label="energy MAE: " + str(self.energiesmae) + " kcal/mol",  alpha=0.3, s=6)
        ax[0].set_xlabel("test")
        ax2.set_ylabel("prediction")
        ax2.legend()
    
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y%m%d-%H%M')
    
        f.suptitle(",".join(species_trained[:3])+"validations", fontsize=14)

        if savefig:
            plt.savefig(self.root + str(self.job_name)+"/" +"&".join(species_trained[:3]) + timestamp + "validation.jpg")

