import torch.nn as nn
import torch.nn.functional as F
import torch

class Attention(nn.Module):
    def __init__(self,dim,num_heads,dropout=0):
        super(Attention, self).__init__()
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

        self.atten = nn.MultiheadAttention(embed_dim=dim,num_heads=num_heads,dropout=dropout)
        self.layer_norm = nn.LayerNorm(dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self,x):
        id = x
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        out , _ =self.atten(q,k,v)

        return out+id


class Encoder_net(nn.Module):
    def __init__(self,dims):
        super(Encoder_net, self).__init__()
        self.layer = nn.Linear(dims[0],dims[1])
        self.attention = Attention(300,4)


    def forward(self,x):
        out1 = self.layer(x)
        out2 = self.attention(out1.unsqueeze(1).squeeze(1))
        out = F.normalize(out2,dim=1,p=2)

        return out

class  reversible_network(nn.Module):
    def __init__(self,dims):
        super(reversible_network, self).__init__()

        self.reduce1 = nn.Linear(dims[0],dims[0]//2)
        self.reduce2 = nn.Linear(dims[0]//2,dims[0])

        self.up1 = nn.Linear(dims[0],dims[0]*2)
        self.up2 = nn.Linear(dims[0]*2,dims[0])

    def forward(self,x,flag):

        if flag:
            reduce_feature = self.reduce2(self.reduce1(x))
            reduce_feature = F.normalize(reduce_feature,dim=1,p=2)
            return reduce_feature
        else:
            up_feature = self.up2(self.up1(x))
            up_feature = F.normalize(up_feature,dim=1,p=2)
            return up_feature

def loss_cal(x,x_aug):
    T = 1.0  # 1.0
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss