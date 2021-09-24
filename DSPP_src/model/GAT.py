import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from torch.nn.modules.module import Module


class GAT(nn.Module):
    def __init__(self, opt):
        super(GAT, self).__init__()
        self.att = Attention(opt)
        self.dropout = opt["dropout"]
        self.leakyrelu = nn.LeakyReLU(opt["leakey"])

    def forward_user(self, ufea, inter, UV_adj):
        learn_user = ufea
        learn_item = inter

        learn_user = F.dropout(learn_user, self.dropout, training=self.training)
        learn_item = F.dropout(learn_item, self.dropout, training=self.training)
        learn_user = self.att(learn_user, learn_item, UV_adj)

        return learn_user

    def forward_item(self, vfea, inter, VU_adj):
        learn_user = inter
        learn_item = vfea

        learn_user = F.dropout(learn_user, self.dropout, training=self.training)
        learn_item = F.dropout(learn_item, self.dropout, training=self.training)
        learn_item = self.att(learn_item, learn_user, VU_adj)

        return learn_item

class Attention(nn.Module):
    def __init__(self,opt):
        super(Attention, self).__init__()
        self.lin = nn.Linear(opt["hidden_dim"], opt["hidden_dim"])

        self.a1 = nn.Parameter(torch.zeros(size=(opt["hidden_dim"], 1)))
        nn.init.xavier_uniform_(self.a1.data, gain=1)
        self.a2 = nn.Parameter(torch.zeros(size=(opt["hidden_dim"], 1)))
        nn.init.xavier_uniform_(self.a2.data, gain=1)

        self.leakyrelu = nn.LeakyReLU(2)
        self.opt = opt
        self.dot = 1
    def forward(self, a, b, adj):
        a = self.lin(a)

        query = a
        key = b

        if self.dot:
            value = self.leakyrelu(
                torch.mm(query, key.transpose(0, 1)) / math.sqrt(self.opt["hidden_dim"]))  # user * item
        else:
            a_input1 = torch.matmul(query, self.a1)
            a_input2 = torch.matmul(key, self.a2)
            value = self.leakyrelu((a_input1 + a_input2.transpose(-1, -2)) / math.sqrt(self.opt["hidden_dim"]))

        zero_vec = -9e15 * torch.ones_like(value)
        value = torch.where(adj.to_dense() > 0, value, zero_vec)
        value = F.softmax(value,dim=1)
        result = torch.matmul(value,key)

        return result