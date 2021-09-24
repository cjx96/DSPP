import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, opt, max_len=4096):
        super().__init__()

        self.opt = opt

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, self.opt["position_dim"]).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, self.opt["position_dim"], 2).float() * -(math.log(10000.0) / self.opt["position_dim"])).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

        if self.cuda:
            self.pe.cuda()

    def forward(self, x):
        return self.pe[x, :]


class BiasedPositionalEmbedding(nn.Module):
    def __init__(self, opt, max_len=4096):
        super().__init__()
        self.opt = opt
        self.sigm = nn.Sigmoid()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, self.opt["biased_position_dim"], 2).float() * -(math.log(10000.0) / self.opt["position_dim"])).exp()
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)
        self.Wt = nn.Linear(1, self.opt["biased_position_dim"] // 2, bias=False)
        if self.cuda:
            self.position.cuda()
            self.div_term.cuda()

    def forward(self, interval):
        phi = self.Wt(interval.unsqueeze(-1))
        phi = self.sigm(phi)
        aa = len(interval.size())
        if aa > 1:
            length = interval.size(1)
        else:
            length = interval.size(0)

        arc = (self.position[:length] * self.div_term).unsqueeze(0)

        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe