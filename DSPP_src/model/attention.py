import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

class multiattention(torch.nn.Module):
    def __init__(self, opt):
        super(multiattention, self).__init__()

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        # self.inputs_size = opt["biased_position_dim"] + opt["original_dim"] + opt["long_dim"]  # pos , orig, long
        self.inputs_size = opt["short_dim"]

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.inputs_size, eps=1e-8)

        self.fc = nn.Linear(opt["short_dim"],opt["original_dim"])
        for i in range(opt["attention_layer"]):
            new_attn_layernorm = torch.nn.LayerNorm(self.inputs_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = MultiHeadedAttention(opt)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.inputs_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.inputs_size, opt["dropout"])
            self.forward_layers.append(new_fwd_layer)

    def forward(self, q, k ,v, attention_mask):
        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](q)
            mha_outputs = self.attention_layers[i](Q, k, v,
                                            att_mask=attention_mask)
            k = k + mha_outputs
            k = self.forward_layernorms[i](k)
            k = self.forward_layers[i](k)
            q = k
            v = k
        log_feats = self.last_layernorm(k)
        return self.fc(log_feats)



class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.dropout1(self.conv1(inputs.transpose(-1, -2)))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs


class MultiHeadedAttention(nn.Module):
    """
    Take in models size and number of heads.
    """
    def __init__(self, opt):
        super().__init__()
        # We assume d_v always equals d_k
        self.opt = opt

        # self.hidden_dim = opt["biased_position_dim"] + opt["original_dim"] + opt["long_dim"]  # pos , orig, long
        self.hidden_dim = opt["short_dim"]

        self.heads = opt["attention_heads"]

        if self.hidden_dim % self.heads:
            print(self.hidden_dim, self.heads)

        self.d_k = self.hidden_dim // self.heads

        assert self.hidden_dim % self.heads == 0

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim, bias=True) for i in range(3)])
        self.output_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=self.opt["dropout"])
        self.activation = nn.ReLU()

    def forward(self, query, key, value, att_mask=None):
        if att_mask is not None:
            # the same mask applies to all heads
            # unsqueeze Returns a new tensor with a dimension of size one
            # inserted at the specified position.
            att_mask = att_mask.unsqueeze(1)

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention.forward(query, key, value, mask=att_mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)

        x = self.output_linear(x)
        return self.activation(x)


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, dropout=None): # 0 means mask it

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        scores = torch.clamp(scores, min=-30, max=30)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores += 1e-5
        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn