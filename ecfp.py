import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter


class ECFP(torch.nn.Module):
    
    def __init__(self, R, Fr, Fe, L, Lp, box_len, device, freeze_fingerprints=False, to_use_xyz=False):
        super(ECFP, self).__init__()
        #self.get_message = [torch.nn.Linear(D_in, D_out) for rad in range(R)]
        #self.get_node_update = [torch.nn.Linear(D_in, D_out) for rad in range(R)]
        self.to_use_xyz = to_use_xyz
        if self.to_use_xyz is True:
            Fe = Fe + 1
        self.box_len = torch.Tensor(box_len).to(device=device.index)
        self.R = R
        self.Lp = Lp # premessage length 
        self.Fr = Fr
        self.Fe = Fe
        self.L = L
        self.F = Fr + Fe
        self.device = device
        #self.H = Parameter(torch.randn(self.R, self.F, self.Fr), requires_grad=not(freeze_fingerprints))
        #self.W = Parameter(torch.randn(self.R+1, self.Fr, self.L), requires_grad=not(freeze_fingerprints))
        
        #self.W = Parameter(torch.randn(R+1, 4, self.L))
        self.W = Parameter(torch.randn(R, Fr, self.L))
        self.K0 = Parameter(torch.randn(R, Fr*2 + Fe, 8))
        self.K1 = Parameter(torch.randn(R, 8, 20))
        self.K2 = Parameter(torch.randn(R, 20, 30))
        self.K3 = Parameter(torch.randn(R, 30, Lp))
        
        self.H0 = Parameter(torch.randn(R, Lp, L))
        self.H1 = Parameter(torch.randn(R, L, Fr))
    
    def get_bond_vector_matrix(self, frame, device, cutoff):
        '''
        input:  xyz torch.Tensor (B, N, 3)
        return:   edge feature matrix torch.Tensor (B, N, N, 3 or 1)
        '''
        box_len = self.box_len
        
        N_atom = frame.shape[1]
        frame = frame.view(-1, N_atom, 1, 3)
        dis_mat = frame.expand(-1, N_atom, N_atom, 3) - frame.expand(-1, N_atom, N_atom, 3).transpose(1,2)
        
        # build minimum image convention 
        mask_pos = dis_mat.ge(0.5*box_len).type(torch.float32).to(device=device.index)
        mask_neg = dis_mat.lt(-0.5*box_len).type(torch.float32).to(device=device.index)       
        # modify distance 
        dis_add = mask_neg * box_len
        dis_sub = mask_pos * box_len
        dis_mat = dis_mat + dis_add - dis_sub
        
        # create cutoff mask
        dis_sq = dis_mat.pow(2).sum(3)                  # compute squared distance of dim (B, N, N)
        mask = (dis_sq <= cutoff ** 2) & (dis_sq != 0)                 # byte tensor of dim (B, N, N)
        A = mask.unsqueeze(3).type(torch.float32).to(device=device.index) #         
        
        # 1) PBC 2) # gradient of zero distance 
        dis_sq = dis_sq.unsqueeze(3)
        dis_sq = dis_sq * A  + 1e-9 # to make sure the distance is not zero, otherwise there will be inf gradient 
        dis_mat = dis_sq.sqrt()
        
        # compute degree of nodes 
        d = A.sum(2).squeeze(2) - 1
        return(dis_mat, A.squeeze(3), d) 
                                
    def forward(self, r, device, xyz=None, cutoff=2.0):
        if (self.to_use_xyz is True) and (xyz is None):
            raise MessageError('xyz must be provided as an argument to ECFP.forward() if you initialized with to_use_xyz=True.')
        if (self.to_use_xyz is False) and (xyz is not None):
            raise MessageError('Why did you provide xyz if to_use_xyz=False?')

        R = self.R  # number of nodes in this graph
        F = self.F  # F = Fr + Fe
        Fr = self.Fr # number of node feature
        Fe = self.Fe # number of edge featue
        L = self.L   # finger print Length
        B = r.shape[0] # batch size
        N = r.shape[1] # number of nodes 

        # Compute the bond_vector_matrix, which has shape (B, N, N, 3), and append it to the edge matrix
        if self.to_use_xyz is True:
            e, A, d= self.get_bond_vector_matrix(frame=xyz, device=self.device, cutoff=cutoff)# .type(torch.float32).to(device=device.index)
            e = e.type(torch.float32).to(device=device.index)
            A = A.type(torch.float32).to(device=device.index)
            d = d.type(torch.long).to(device=device.index)
    
        # Check for dimensional consistency
        if r.shape[2] != Fr:
            raise MessageError("Inconsistent atom feature vector length (Fr).")
        if e.shape[3] != Fe:
            raise MessageError("Inconsistent bond feature vector length (Fe).")
        if not(e.shape[0] == A.shape[0] == d.shape[0] == r.shape[0] == B):
            raise MessageError("Inconsistent batch size (B).")
        if not(e.shape[1] == e.shape[2] == A.shape[1] == A.shape[2] == d.shape[1]== r.shape[1]==N):
            raise MessageError("Inconsistent number of atoms (N).")
        
        relu = torch.nn.functional.relu            
        softmax = torch.nn.Softmax(dim=2)
        sigmoid = torch.nn.Sigmoid() 
        tanh = torch.nn.Tanh()
        elu = torch.nn.ELU()

        # Initialize the fingerprint tensor (f)
        f = torch.zeros(B, self.L).type(torch.float32).to(device=device.index)
        # Iteratively update the feature tensor (r) and the fingerprint tensor (f)  
        W = self.W
        K0 = self.K0
        K1 = self.K1
        K2 = self.K2
        K3 = self.K3
        K4 = self.K4
        K5 = self.K5
        
        H0 = self.H0
        H1 = self.H1
        
        for rad in range(R):

            f = f + sigmoid(r.matmul(W[rad])).sum(1)
            
            # concatenate node and edge this is to build a matrix of dimension of (B, N, N, 2*Fr + Fe)
            # maybe a separate function?
            node_cat = torch.cat((r.unsqueeze(1).expand(-1, N, N, 1) , r.unsqueeze(2).expand(-1, N, N, 1)) , dim =3)
            node_edge_cat = torch.cat((node_cat, e), dim=3)
            
            # construct premessage 
            premessage = node_edge_cat * (A.unsqueeze(3).expand_as(node_edge_cat)) # only keep relavant edge 
            
            # feed premessage into the neural network
            premessage = premessage.matmul(K0[rad])
            premessage = tanh(premessage)
            premessage = premessage.matmul(K1[rad])
            premessage = tanh(premessage)
            premessage = premessage.matmul(K2[rad])
            premessage = tanh(premessage)
            premessage = premessage.matmul(K3[rad])
            
            # sum to get message 
            message = premessage.sum(1)
            # message neural network 
            message = message.matmul(H0[rad])
            message = tanh(message)
            message = message.matmul(H1[rad])
            r = message.reshape(B, N, Fr)

        f = f + sigmoid(r.matmul(W[rad])).sum(1)
        return(f)