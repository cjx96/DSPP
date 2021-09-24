import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import scipy.sparse as sp
from utils import torch_utils
from model.DSPP import DSPP
import pdb
import time


class Trainer(object):
    def __init__(self, opt):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):  # here should change
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch=0):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


class DSPPTrainer(Trainer):
    def __init__(self, opt):
        self.opt = opt
        self.model = DSPP(opt)

        self.user_long_embedding_prev = nn.Embedding(opt["num_user"], opt["long_dim"])
        self.item_long_embedding_prev = nn.Embedding(opt["num_item"], opt["long_dim"])

        self.MSEcriterion = nn.MSELoss()
        self.SmoothMSEcriterion = nn.MSELoss()
        self.BCE = nn.BCEWithLogitsLoss()

        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.model.parameters(), opt['lr'],
                                                   l2=opt['weight_decay'])

        self.pred_loss = 0
        self.mse_loss = 0
        self.rank_loss = 0

        self.to_cuda()

    def to_cuda(self):
        if self.opt['cuda']:
            self.model.cuda()
            self.MSEcriterion.cuda()
            self.SmoothMSEcriterion.cuda()
            self.BCE.cuda()
            self.model.user_index = self.model.user_index.cuda()
            self.model.item_index = self.model.item_index.cuda()
            self.user_long_embedding_prev = self.user_long_embedding_prev.cuda()
            self.item_long_embedding_prev = self.item_long_embedding_prev.cuda()

    def init_embedding(self):
        nn.init.xavier_normal_(self.user_long_embedding_prev.weight, gain=1.0)
        nn.init.xavier_normal_(self.item_long_embedding_prev.weight, gain=1.0)

    def init_embedding(self):
        nn.init.xavier_normal_(self.user_long_embedding_prev.weight, gain=1.0)
        nn.init.xavier_normal_(self.item_long_embedding_prev.weight, gain=1.0)
        self.pred_loss = 0
        self.mse_loss = 0
        self.rank_loss = 0

    def change_optimizer(self, cur_lr):
        self.optimizer = torch_utils.get_optimizer(self.opt['optim'], self.model.parameters(), cur_lr,
                                                   l2=self.opt['weight_decay'])

    def get_score(self, user_embedding, item_embedding):
        if self.opt["metric"] == "nn":
            return self.get_nn_score(user_embedding, item_embedding)
        if self.opt["metric"] == "dot":
            return self.get_dot_score(user_embedding, item_embedding)
        if self.opt["metric"] == "mse":
            return self.get_MSE_score(user_embedding, item_embedding)
        if self.opt["metric"] == "bce":
            return self.BCE(user_embedding, item_embedding)

    def get_nn_score(self, user_embedding, item_embedding):
        user_item_embedding = torch.cat((user_embedding, item_embedding), dim=-1)
        predict_score = self.model.score_function(user_item_embedding)
        # predict_score = F.sigmoid(predict_score)
        return predict_score.squeeze(-1)

    def get_dot_score(self, user_embedding, item_embedding):
        output = (user_embedding * item_embedding).sum(dim=-1)
        # output = F.sigmoid(output)
        return output

    def get_MSE_score(self, user_embedding, item_embedding):
        predict_score = self.MSEcriterion(user_embedding, item_embedding.detach())
        return predict_score

    def HingeLoss(self, pos, neg, gap=-1):
        pos = F.sigmoid(pos)
        neg = F.sigmoid(neg)
        if gap < 0:
            gamma = torch.tensor(0.1)
        else:
            gamma = torch.tensor(gap)
        if self.opt["cuda"]:
            gamma = gamma.cuda()
        return F.relu(gamma - pos + neg).mean()

    def unpack_batch(self, batch):
        userid = batch["user"]
        itemid = batch["item"]
        pred_time = batch["pred_time"]
        attention_window = batch["attention_window"]
        timestamp_window = batch["timestamp_window"]
        pos_itemid = batch["pos_itemid"]
        neg_itemid = batch["neg_itemid"]
        return userid, itemid, pred_time, attention_window, timestamp_window, pos_itemid, neg_itemid

    def update_graph_prev(self, graph):
        user_feature = self.model.user_embedding(self.model.user_index)
        item_feature = self.model.item_embedding(self.model.item_index)
        user_feature = user_feature.detach()
        item_feature = item_feature.detach()
        user_long_feature_prev = self.user_long_embedding_prev(self.model.user_index)
        item_long_feature_prev = self.item_long_embedding_prev(self.model.item_index)

        if graph is None:
            user_long_feature = user_feature
            item_long_feature = item_feature
        else:
            if self.opt["cuda"]:
                UV_adj = graph.UV_adj.cuda()
                VU_adj = graph.VU_adj.cuda()
                # adj = graph.all_adj.cuda()
            else:
                UV_adj = graph.UV_adj
                VU_adj = graph.VU_adj
                # adj = graph.all_adj
            user_long_feature, item_long_feature = self.model.GNN(user_feature, item_feature, UV_adj, VU_adj, adj=None)

        user_long_feature = \
        self.model.GRU_user_long(user_long_feature.unsqueeze(1), user_long_feature_prev.unsqueeze(0))[0].squeeze(1)
        item_long_feature = \
        self.model.GRU_item_long(item_long_feature.unsqueeze(1), item_long_feature_prev.unsqueeze(0))[0].squeeze(1)
        self.user_long_embedding_prev.weight.data.copy_(user_long_feature)
        self.item_long_embedding_prev.weight.data.copy_(item_long_feature)

    def my_index_select(self, memory, index):
        tmp = list(index.size()) + [-1]
        # import pdb
        # pdb.set_trace()
        index = index.view(-1)
        ans = torch.index_select(memory, 0, index)
        ans = ans.view(tmp)
        return ans

    def get_long_preference(self, graph, userid, long_pos_itemid, long_neg_itemid, evaluate=False):
        user_feature = self.model.user_embedding(self.model.user_index)
        item_feature = self.model.item_embedding(self.model.item_index)
        user_long_feature_prev = self.user_long_embedding_prev(self.model.user_index)
        item_long_feature_prev = self.item_long_embedding_prev(self.model.item_index)

        if graph is None:
            user_long_feature = user_feature
            item_long_feature = item_feature
        else:
            if self.opt["cuda"]:
                UV_adj = graph.UV_adj.cuda()
                VU_adj = graph.VU_adj.cuda()
                # adj = graph.all_adj.cuda()
            else:
                UV_adj = graph.UV_adj
                VU_adj = graph.VU_adj
                # adj = graph.all_adj
            user_long_feature, item_long_feature = self.model.GNN(user_feature, item_feature, UV_adj, VU_adj, adj=None)

        user_long_feature = \
        self.model.GRU_user_long(user_long_feature.unsqueeze(1), user_long_feature_prev.unsqueeze(0))[0].squeeze(1)
        item_long_feature = \
        self.model.GRU_item_long(item_long_feature.unsqueeze(1), item_long_feature_prev.unsqueeze(0))[0].squeeze(1)
        self.user_long_embedding = user_long_feature
        self.item_long_embedding = item_long_feature

        if evaluate:
            return 0

        user_long_preference = self.my_index_select(user_long_feature, userid)
        item_long_pos_preference = self.my_index_select(item_long_feature, long_pos_itemid)
        item_long_neg_preference = self.my_index_select(item_long_feature, long_neg_itemid)

        long_positive_score = self.get_dot_score(user_long_preference.unsqueeze(1), item_long_pos_preference)
        long_negative_score = self.get_dot_score(user_long_preference.unsqueeze(1), item_long_neg_preference)

        long_loss = self.HingeLoss(long_positive_score, long_negative_score, 0.3)
        return long_loss

    def get_adapt_sequence_embedding(self, user_id, attention_window, timestamp_window):
        userid_orig = self.model.user_embedding(user_id)  # batch_size * hidden
        itemid_orig = self.model.item_embedding(attention_window)  # batch_size * seqs * hidden
        bias_position_emb = self.model.time_aware_position_embedding(timestamp_window)  # torch.Size([25, 10, 32])

        userid_orig = userid_orig.unsqueeze(1).repeat(1, itemid_orig.size(1), 1)
        user_adapt = self.model.lin_Q(torch.cat((userid_orig, itemid_orig + bias_position_emb), dim=-1))
        item_adapt = itemid_orig + bias_position_emb
        batch_size = item_adapt.size()[0]
        seqs_length = item_adapt.size()[1]
        attention_mask = torch.tril(torch.ones((batch_size, seqs_length, seqs_length), dtype=torch.float))
        if self.opt["cuda"]:
            attention_mask = attention_mask.cuda()
        sequence_embedding = self.model.attention(user_adapt, item_adapt, item_adapt,
                                                  attention_mask)  # batch_size * seqs * hidden
        return sequence_embedding

    def forward_batch(self, batch, prev_graph, evaluate=False):
        userid, itemid, pred_time, attention_window, timestamp_window, pos_itemid, neg_itemid = self.unpack_batch(
            batch)

        if evaluate:
            self.model.eval()
        else:
            self.model.train()
            self.optimizer.zero_grad()

        long_loss = self.get_long_preference(prev_graph, userid, pos_itemid, neg_itemid[:, :, 1], evaluate)

        sequence_embedding = self.get_adapt_sequence_embedding(userid, attention_window,
                                                               timestamp_window)  # batch_size * seqs * original_dim
        user_short_prev_o = self.model.user_embedding(userid).unsqueeze(1).repeat(1, sequence_embedding.size(1),
                                                                                  1)  # batch_size * seqs * original_dim
        item_short_prev_o = self.model.item_embedding(attention_window)

        # pdb.set_trace()
        user_short = self.model.user_update(sequence_embedding.view(-1, 1, self.opt["short_dim"]),
                                            user_short_prev_o.view(1, -1, self.opt["short_dim"]).detach())[0].view_as(
            sequence_embedding)  # batch_size * seqs * original_dim
        item_short = self.model.item_update(sequence_embedding.view(-1, 1, self.opt["short_dim"]),
                                            item_short_prev_o.view(1, -1, self.opt["short_dim"]).detach())[0].view_as(
            sequence_embedding)  # batch_size * seqs * original_dim

        cal_latest = self.opt["attention_window"] // 3
        mseloss = self.MSEcriterion(user_short_prev_o[:, -cal_latest:], user_short[:, -cal_latest:])  # using the BP algorithm to approximate the update embedding
        mseloss += self.MSEcriterion(item_short_prev_o[:, -cal_latest:], item_short[:, -cal_latest:])

        user_short_pred = self.model.user_shift(userid.unsqueeze(1).repeat(1, user_short.size(1)), user_short, pred_time) # batch_size * seq * original_dim
        user_long = self.my_index_select(self.user_long_embedding, userid)  # batch_size * original_dim
        user_long = user_long.unsqueeze(1).repeat(1, sequence_embedding.size(1), 1) # batch_size * seq * original_dim
        user_total = torch.cat((user_long, user_short_pred), dim=-1)  # batch_size * seqs * 2*original_dim

        if evaluate:
            rank = []
            all_item_long_preference = self.my_index_select(self.item_long_embedding, self.model.item_index)
            all_item_short_preference = self.model.item_embedding(self.model.item_index)

            for i in range(len(userid)):
                cur_user_embedding = user_total[i, -1]
                cur_user_embedding = cur_user_embedding.repeat(self.opt["num_item"], 1)

                item_short = self.model.item_shift(self.model.item_index.unsqueeze(-1), all_item_short_preference.unsqueeze(1), pred_time[i,-1:].unsqueeze(1).repeat(self.opt["num_item"],1)).squeeze(1)
                item_total = torch.cat((all_item_long_preference, item_short), dim=-1)  # batch_size * seqs * 2*original_dim

                all_lambda = self.get_score(cur_user_embedding, item_total)  # calculating the 'intensity function' rank of target item
                cur_lambda = all_lambda[itemid[i]]

                score_larger = (all_lambda > (cur_lambda + 0.00001)).data.cpu().numpy()
                true_item_rank = max(np.sum(score_larger) + 1, 1)
                rank.append(true_item_rank)
            return rank

        pos_item_long = self.my_index_select(self.item_long_embedding, pos_itemid)  # batch_size * seqs * original_dim
        pos_item_short = self.model.item_embedding(pos_itemid)  # batch_size * seqs * original_dim
        pos_item_short_pred = self.model.item_shift(pos_itemid, pos_item_short, pred_time)
        pos_item_total = torch.cat((pos_item_long, pos_item_short_pred), dim=-1)  # batch_size * seqs * 2*original_dim

        neg_item_long = self.my_index_select(self.item_long_embedding,
                                             neg_itemid)  # batch_size * seqs * numsample * original_dim
        neg_item_short = self.model.item_embedding(neg_itemid)  # batch_size * seqs * numsample * original_dim
        neg_time = pred_time.unsqueeze(1).repeat(1,neg_itemid.size(2),1).view(pos_item_short.size(0),-1)
        neg_time_monte = neg_time * torch.randint(1,65, neg_time.size()).float().cuda(neg_time.device)/64 # monte carlo estimate
        neg_item_short_pred = self.model.item_shift(neg_itemid.view(pos_item_short.size(0),-1), neg_item_short.view(pos_item_short.size(0),-1,pos_item_short.size(2)), neg_time_monte).view_as(neg_item_short)
        neg_item_total = torch.cat((neg_item_long, neg_item_short_pred),dim=-1)  # batch_size * seqs * numsample * (2*original_dim)

        positive_lambda = self.get_score(user_total[:, -cal_latest:], pos_item_total[:, -cal_latest:])
        negative_lambda = self.get_score(
            (user_total[:, -cal_latest:]).unsqueeze(-2).repeat(1, 1, neg_item_total.size()[2], 1),
            neg_item_total[:, -cal_latest:])

        positive_lambda = positive_lambda.unsqueeze(-1).repeat(1, 1, self.opt["negative_sample"])
        pos_labels, neg_labels = torch.ones(positive_lambda.size()), torch.zeros(
            negative_lambda.size())

        if self.opt["cuda"]:
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()
        predloss = self.BCE(positive_lambda, pos_labels)  # maximizing the intensity function of positive samples
        predloss += self.BCE(negative_lambda, neg_labels)  # minimizing the intensity function of negative samples

        self.pred_loss += predloss.item()
        self.mse_loss += mseloss.item()

        loss = predloss + long_loss + 0.1 * mseloss

        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss.item()