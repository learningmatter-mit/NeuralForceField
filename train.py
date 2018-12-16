import json
import numpy as np
import torch

from util import Data

import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
from schnet import *
import matplotlib.pyplot as plt
import argparse

#----------------Load parameter file----------------
parser = argparse.ArgumentParser()
parser.add_argument('json')
args = parser.parse_args()
par = json.load(open(args.json, "r"))

N_data = par["N_data"]
train_percent = par["train_percent"]
T = par["T"]
batch_size = par["batch_size"]
data = par["data"]
lr = par["lr"]
n_basis = par["n_basis"]
period = par["period"]
epoch = par["epoch"]
cutoff_soft = par["cutoff_soft"]
log_dir = par["log_dir"]
gamma = par["gamma"]
n_gaussians = par['n_gaussians']

# Load Data 
device = 0
AtomData = Data(par)

n_filters = n_basis
n_atom_basis = n_basis

model = Net(n_atom_basis=n_atom_basis, n_filters=n_filters,
			 n_gaussians=n_gaussians, cutoff_soft= cutoff_soft, device=device, T=T).to(device)

# training/test loss log
test_u_log = []
train_u_log = []
test_f_log = []
train_f_log = []

# set up optimizer
optimizer = optim.Adam(list(model.parameters()), lr=1e-4) 
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=period, gamma=gamma)

criterion = torch.nn.MSELoss()
mae = torch.nn.L1Loss()

for epoch in range(5000):
    # train 
    train_u_mae = 0.0
    train_force_mae = 0.0
    
    scheduler.step()
    
    for i in range(AtomData.xyz_train.shape[0]):
        
        node = Variable(torch.Tensor(AtomData.node_train)).cuda(device)
        xyz = Variable(torch.Tensor(AtomData.xyz_train[i])).cuda(device)
        xyz.requires_grad = True
        force = torch.Tensor(AtomData.force_train[i]).cuda(device)
        energy = torch.Tensor(AtomData.energy_train[i]).cuda(device)
        
        U = model(r=node, xyz=xyz)
        f_pred = -compute_grad(inputs=xyz, output=U)
        
        # comput eloss
        loss_force = criterion(f_pred, force)
        loss_u = criterion(U, energy)
        loss =  loss_force + 0.01*loss_u
        
        # update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # compute MAE
        train_u_mae += mae(U, energy) # compute MAE
        train_force_mae += mae(f_pred, force)
        
    # averaging MAE
    train_u = train_u_mae.data[0]/(AtomData.xyz_train.shape[0])
    train_force = train_force_mae.data[0]/(AtomData.xyz_train.shape[0])
    
    train_u_log.append(train_u)
    train_f_log.append(train_force)
    # test 
    
    test_u_mae = 0.0
    test_force_mae = 0.0
    
    del xyz, force, energy

    with torch.no_grad():
        for i in range(AtomData.xyz_test.shape[0]):

            node = Variable(torch.Tensor(AtomData.node_test), volatile=True).cuda(device)
            xyz = Variable(torch.Tensor(AtomData.xyz_test[i])).cuda(device)
            xyz.requires_grad = True
            force = torch.Tensor(AtomData.force_test[i]).cuda(device)
            energy = torch.Tensor(AtomData.energy_test[i]).cuda(device)

            U = model(r=node, xyz=xyz)
            #f_pred = -compute_grad(inputs=xyz, output=U)

            # loss
            test_u_mae += mae(U, energy) # compute MAE
            #test_force_mae += mae(f_pred, force)
        
    # averaging MAE
    test_u = test_u_mae.data[0]/(AtomData.xyz_test.shape[0])
    test_force = 0.0 #test_force_mae.data[0]/(xyz_test.shape[0])
    
    test_u_log.append(test_u)
    test_f_log.append(test_force)
    
    # check for convergence
    #if np.abs((np.array(test_u_log[-9: -5]).mean() - np.array(test_u_log[-5: -1:])).mean())/np.array(test_u_log[-9: -5]).mean() < 1e-4:
    #    print("converged!")
    #    break

    print("epoch %d  force train:  %.3f  U train: %.3f  force test:%.3f  U test: %.3f" % 
          (epoch, train_force, train_u, test_force, test_u))

# save train_log energy 
u_pred = []
for i, xyz in enumerate(AtomData.xyz_test):

    node = torch.Tensor(AtomData.node_test).to(device)
    xyz = torch.Tensor(xyz).to(device)
    energy = torch.Tensor(AtomData.energy_test[i]).to(device)
    U = model(r=node, xyz=xyz)
    u_pred.append(U.detach().cpu().numpy())
    
u_pred = np.array(u_pred).reshape(-1)
u_true = AtomData.energy_test.detach().numpy().reshape(-1)
test_u_log = np.array(test_u_log)
train_u_log = np.array(train_u_log)


import os 
os.mkdir(log_dir)
np.savetxt(log_dir + "u_pred.txt", u_pred) 
np.savetxt(log_dir + "u_true.txt", u_true)
np.savetxt(log_dir + "test_log.txt", test_u_log)
np.savetxt(log_dir + "train_log.txt", train_u_log)