import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn import Tanh
from torch.nn import ReLU

class resblock(nn.Module):
    def __init__(self, in_dim, width):
        super(resblock, self).__init__()
        self.layer1 = nn.Linear(in_dim, width)
        self.layer2 = nn.Linear(width, in_dim)

    def forward(self,x):
        
        residual = x
        
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        
        out += residual
        return out
    
class linearblock(nn.Module):
    def __init__(self, in_dim, width):
        super(linearblock, self).__init__()
        self.layer1 = nn.Linear(in_dim, width)
        self.layer2 = nn.Linear(width, in_dim)

    def forward(self,x):
        
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        
        return out

class GraphDis(torch.nn.Module):
    
    def __init__(self, Fr, Fe, device, cutoff, pbc=False, box_len=None):
        super(GraphDis, self).__init__()
        #self.get_message = [torch.nn.Linear(D_in, D_out) for rad in range(R)]
        #self.get_node_update = [torch.nn.Linear(D_in, D_out) for rad in range(R)]

        self.Fr = Fr
        self.Fe = Fe # include distance
        self.F = Fr + Fe
        self.device = device
        self.cutoff = cutoff
        self.pbc = pbc

        if pbc == True and box_len==None:
            raise RuntimeError("require box vector input!")
        if pbc:
            self.box_len = torch.Tensor(box_len).cuda(self.device)
    
    def get_bond_vector_matrix(self, frame):
        '''
        input:  xyz torch.Tensor (B, N, 3)
        return:   edge feature matrix torch.Tensor (B, N, N, 3 or 1)
        '''
        device = self.device
        cutoff = self.cutoff
        
        N_atom = frame.shape[1]
        frame = frame.view(-1, N_atom, 1, 3)
        dis_mat = frame.expand(-1, N_atom, N_atom, 3) - frame.expand(-1, N_atom, N_atom, 3).transpose(1,2)
        
        if self.pbc:

            # build minimum image convention 
            box_len = self.box_len
            mask_pos = dis_mat.ge(0.5*box_len).type(torch.FloatTensor).cuda(self.device)
            mask_neg = dis_mat.lt(-0.5*box_len).type(torch.FloatTensor).cuda(self.device)
            
            # modify distance 
            dis_add = mask_neg * box_len
            dis_sub = mask_pos * box_len
            dis_mat = dis_mat + dis_add - dis_sub
        
        # create cutoff mask
        dis_sq = dis_mat.pow(2).sum(3)                  # compute squared distance of dim (B, N, N)
        mask = (dis_sq <= cutoff ** 2) & (dis_sq != 0)                 # byte tensor of dim (B, N, N)
        A = mask.unsqueeze(3).type(torch.FloatTensor).cuda(self.device) #         
        
        # 1) PBC 2) # gradient of zero distance 
        dis_sq = dis_sq.unsqueeze(3)
        dis_sq = (dis_sq * A) #+ 1e-8# to make sure the distance is not zero, otherwise there will be inf gradient 
        dis_mat = dis_sq#.sqrt()
        
        # compute degree of nodes 
        d = A.sum(2).squeeze(2) - 1
        return(dis_mat, A.squeeze(3), d) 
                                
    def forward(self, r, xyz=None):

        F = self.F  # F = Fr + Fe
        Fr = self.Fr # number of node feature
        Fe = self.Fe # number of edge featue
        B = r.shape[0] # batch size
        N = r.shape[1] # number of nodes 
        device = self.device

        # Compute the bond_vector_matrix, which has shape (B, N, N, 3), and append it to the edge matrix
        e, A, d= self.get_bond_vector_matrix(frame=xyz)# .type(torch).to(device=device.index)
        e = e.type(torch.FloatTensor).cuda(self.device)
        A = A.type(torch.FloatTensor).cuda(self.device)
        d = d.type(torch.LongTensor).cuda(self.device)
        
        return(r, e, A)
    
class GraphConv(torch.nn.Module):
    def __init__(self, in_channel,
                       out_channel, 
                       mid_channel,
                       Fe):
        super(GraphConv, self).__init__()
        
        self.in_channel = in_channel # premessage length 
        self.mid_channel = mid_channel
        self.out_channel = out_channel 
        self.Fe = Fe

        tanh = torch.nn.Tanh()
        relu = torch.nn.ReLU()
        # input dimension Fr*2 + Fe, output dimension LP
        self.K_net = nn.Sequential(  
                     nn.Linear(in_channel*2 + Fe, 8),
                     nn.Linear(8, 8),
                     linearblock(8, 8),
                     linearblock(8, 8),
                     nn.Linear(8, 8),
                     ReLU(),
                     nn.Linear(8, 8),
                     ReLU(),
                     nn.Linear(8, mid_channel))        
        # input dimension Lp, output dimension Fr
        self.H_net = nn.Sequential(  
                     nn.Linear(mid_channel, 8),
                     relu,
                     linearblock(8, 8),
                     linearblock(8, 8),
                     nn.Linear(8, out_channel))
                                
    def forward(self, r, e, A):

        Fe = self.Fe # number of edge featue
        B = r.shape[0] # batch size
        N = r.shape[1] # number of nodes 
    
        # Check for dimensional consistency
        if r.shape[2] != self.in_channel:
            raise MessageError("Inconsistent atom feature vector length (Fr).")
        if e.shape[3] != Fe:
            raise MessageError("Inconsistent bond feature vector length (Fe).")
        if not(e.shape[0] == A.shape[0] ==r.shape[0] == B):
            raise MessageError("Inconsistent batch size (B).")
        if not(e.shape[1] == e.shape[2] == A.shape[1] == A.shape[2]== r.shape[1]==N):
            raise MessageError("Inconsistent number of atoms (N).")
        
        relu = torch.nn.functional.relu            
        softmax = torch.nn.Softmax(dim=2)
        sigmoid = torch.nn.Sigmoid() 
        tanh = torch.nn.Tanh()
            
        # concatenate node and edge this is to build a matrix of dimension of (B, N, N, 2*Fr + Fe)
        # maybe a separate function?
        node_cat = torch.cat((r.unsqueeze(1).expand(-1, N, N, self.in_channel) , r.unsqueeze(2).expand(-1, N, N, self.in_channel)) , dim =3)
        node_edge_cat = torch.cat((node_cat, e), dim=3)
        # construct premessage 
        premessage = node_edge_cat * (A.unsqueeze(3).expand_as(node_edge_cat))
        #self.premessage = premessage 
        # only keep relavant edge 
        # feed premessage into the neural network
        premessage = self.K_net(premessage)
        # sum to get message 
        message = premessage.sum(1)
        # message neural network 
        message = self.H_net(message)
        
        r = message.view(B, N, self.out_channel)
        #r = message.reshape(B, N, self.out_channel)
        return(r)

class FP(torch.nn.Module):
    def __init__(self, L, Fr):
        super(FP, self).__init__()
        tanh = torch.nn.Tanh()
        relu = torch.nn.ReLU()
        self.F_net = nn.Sequential(  
                     nn.Linear(Fr, L),
                     relu)
        
    def forward(self, r):

        f = self.F_net(r).sum(1)
        return(f)


    
class Model(torch.nn.Module):
    def __init__(self, L, device, radius, cutoff):
        '''
        L: finger print length
        
        device: device used "cuda:0" or "cuda:1"
        
        cnn: if True using convolution, if False, using NN 
        
        cutoff: node distance cutoff, default 0.2
        '''
        
        super(Model, self).__init__()
        self.radius = radius
        self.cutoff = cutoff
        
        self.graph_dis = GraphDis(Fr=1, Fe=1, cutoff=cutoff, device=device)
        self.FP1 = FP(L = L, Fr = 1)
        
        if radius >= 1:
            self.conv1 = GraphConv(in_channel = 1, Fe=1, out_channel=8, mid_channel = 8)
            self.FP2 = FP(L = L, Fr = 8)
            
        if radius >= 2:
            self.conv2 = GraphConv(in_channel = 8, Fe=1, out_channel=8, mid_channel = 8)
            self.FP3 = FP(L = L, Fr = 8)
            
        if radius >= 3:
            self.conv3 = GraphConv(in_channel = 8, Fe=1, out_channel=8, mid_channel = 8)
            self.FP4 = FP(L = L, Fr = 8)
        
        # nueral nets
        self.NN =  nn.Sequential(
                     nn.Linear(L, 32),
                     linearblock(32,32),
                     nn.Linear(32, 16),
                     ReLU(),
                     nn.Linear(16, 1))
        
    def forward(self, r, xyz):
        
        r, e, A = self.graph_dis(xyz=xyz, r=r)
        
        
        if self.radius == 1:
            f = self.FP1(r)
            r = self.conv1(r, e, A)
            f += self.FP2(r)
            
        elif self.radius == 2:
            f = self.FP1(r)
            r = self.conv1(r, e, A)
            f += self.FP2(r)
            r = self.conv2(r, e, A)
            f += self.FP3(r)   
            
        elif self.radius == 3:
            f = self.FP1(r)
            r = self.conv1(r, e, A)
            f += self.FP2(r)
            r = self.conv2(r, e, A)
            f += self.FP3(r)
            r = self.conv3(r, e, A)
            f += self.FP4(r)
            
        u = self.NN(f)
        return(u)
