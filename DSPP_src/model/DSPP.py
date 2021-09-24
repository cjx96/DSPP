import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.GNN import GNN
from torch.autograd import Variable
from model.attention import multiattention
from model.embedding import PositionalEmbedding,BiasedPositionalEmbedding

class DSPP(nn.Module):
    def __init__(self, opt):
        super(DSPP, self).__init__()
        print("*** Initializing the DSPP model ***")
        self.opt = opt

        self.GNN = GNN(opt)
        self.attention = multiattention(opt)

        self.GRU_user_long = nn.GRU(input_size = opt["original_dim"], hidden_size = opt["original_dim"], batch_first=True)
        self.GRU_item_long = nn.GRU(input_size = opt["original_dim"], hidden_size = opt["original_dim"], batch_first=True)

        self.user_update = nn.GRU(input_size = opt["original_dim"], hidden_size = opt["original_dim"], batch_first=True)
        self.item_update = nn.GRU(input_size = opt["original_dim"], hidden_size = opt["original_dim"], batch_first=True)

        self.lin_Q = nn.Linear(2*opt["original_dim"], opt["original_dim"])

        self.user_embedding = nn.Embedding(opt["num_user"], opt["original_dim"])
        self.item_embedding = nn.Embedding(opt["num_item"], opt["original_dim"])
        
        self.time_aware_position_embedding = BiasedPositionalEmbedding(opt)

        self.score_function = nn.Sequential(nn.Linear(4*opt["original_dim"], 10),
                                            nn.ReLU(),
                                            nn.Linear(10, 1)
                                            )

        self.user_index = torch.arange(0, self.opt["num_user"], 1)
        self.item_index = torch.arange(0, self.opt["num_item"], 1)

        self.user_shift_embedding = nn.Embedding(opt["num_user"], opt["original_dim"])
        self.item_shift_embedding = nn.Embedding(opt["num_item"], opt["original_dim"])
        print("*** DSPP initialization complete ***\n\n")

    def user_shift(self, id, emb, time_gap):
        shift_emb = self.user_shift_embedding(id)  # batch_size * seq * embedding_size
        time_gap = time_gap.unsqueeze(-1).repeat(1, 1, shift_emb.size(2)) # time_gap: batch_size * seq * embedding_size
        after_shift_embedding = emb * (1 + F.sigmoid(time_gap * shift_emb))
        return after_shift_embedding

    def item_shift(self, id, emb, time_gap):
        shift_emb = self.item_shift_embedding(id)  # batch_size * seq * embedding_size
        time_gap = time_gap.unsqueeze(-1).repeat(1, 1, shift_emb.size(2)) # time_gap: batch_size * seq * embedding_size
        after_shift_embedding = emb * (1 + F.sigmoid(time_gap * shift_emb))
        return after_shift_embedding