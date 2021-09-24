import numpy as np
import scipy.sparse as sp
import torch
import warnings
import random

warnings.filterwarnings('ignore')

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class BipartiteGraph(object):
    def __init__(self, opt):
        self.opt = opt
        self.num_user = self.opt["num_user"]
        self.num_item = self.opt["num_item"]
        self.reset()

    def reset(self):
        self.UV_edges = []
        self.VU_edges = []
        self.edges = []

        self.UV_weight = []
        self.VU_weight = []
        self.weight = []

        self.bigraph = {}
        self.graph_interaction_number = 0

    def add_pre_graph(self, pre_graph, pre_number, decay):
        # self.graph_interaction_number += pre_number
        self.graph_interaction_number = 0
        for user in pre_graph.keys():
            if user not in self.bigraph:
                self.bigraph[user]={}
            for item in pre_graph.keys():
                if item not in self.bigraph[user]:
                    self.bigraph[user][item] = 0
                self.bigraph[user][item] += pre_graph[user][item] / decay

    def add_edge(self, user, item):
        if user not in self.bigraph:
            self.bigraph[user] = {}
        if item not in self.bigraph[user]:
            self.bigraph[user][item] = 0
        self.bigraph[user][item] += 1
        self.graph_interaction_number += 1

    def random_neg(self,user):
        if user not in self.bigraph:
            return -1
        n = 100
        rand = random.randint(0, self.opt["num_item"] - 2) # del pad
        while n:
            n-=1
            rand = random.randint(0, self.opt["num_item"] - 2) # del pad
            if rand not in self.bigraph[user]:
                return rand
        return rand

    def random_pos(self,user):
        if user not in self.bigraph:
            return -1
        rand = random.randint(0, self.opt["num_item"] - 2) # del pad
        rand %= len(list(self.bigraph[user].keys()))
        return list(self.bigraph[user].keys())[rand]

    def random_list_neg(self,user, positive_list):
        ret = []
        for _ in positive_list:
            ret.append(self.random_neg(user))
        return ret

    def random_list_pos(self,user, positive_list):
        ret = []
        for _ in positive_list:
            ret.append(self.random_pos(user))
        return ret

    def random_list_multi_neg(self, user, positive_list, nega_number):
        ret = []
        for _ in positive_list:
            res = []
            for i in range(nega_number):
                res.append(self.random_neg(user))
            ret.append(res)
        return ret

    def change_to_tensor(self, self_loop = 0):
        self.UV_edges = []
        self.VU_edges = []
        self.edges = []

        self.UV_weight = []
        self.VU_weight = []
        self.weight = []
        for user in self.bigraph.keys():
            for item in self.bigraph[user].keys():
                self.UV_edges.append([user, item])
                self.UV_weight.append(self.bigraph[user][item])

                self.VU_edges.append([item, user])
                self.VU_weight.append(self.bigraph[user][item])

                self.edges.append([user, item + self.opt["num_user"]])
                self.edges.append([item + self.opt["num_user"], user])
                self.weight.append(self.bigraph[user][item])
                self.weight.append(self.bigraph[user][item])

        self.UV_edges = np.array(self.UV_edges)
        self.VU_edges = np.array(self.VU_edges)
        self.all_edges = np.array(self.edges)

        if self.opt["weight"]:
            # print(self.UV_weight)
            # print(self.VU_weight)
            # print(self.weight)
            UV_adj = sp.coo_matrix((np.array(self.UV_weight), (self.UV_edges[:, 0], self.UV_edges[:, 1])),
                                   shape=(self.opt["num_user"], self.opt["num_item"]),
                                   dtype=np.float32)
            VU_adj = sp.coo_matrix((np.array(self.VU_weight), (self.VU_edges[:, 0], self.VU_edges[:, 1])),
                                   shape=(self.opt["num_item"], self.opt["num_user"]),
                                   dtype=np.float32)
            all_adj = sp.coo_matrix((np.array(self.weight), (self.all_edges[:, 0], self.all_edges[:, 1])), shape=(
                self.opt["num_item"] + self.opt["num_user"], self.opt["num_item"] + self.opt["num_user"]),
                                    dtype=np.float32)
        else :
            UV_adj = sp.coo_matrix((np.ones(self.UV_edges.shape[0]), (self.UV_edges[:, 0], self.UV_edges[:, 1])),
                                   shape=(self.opt["num_user"], self.opt["num_item"]),
                                   dtype=np.float32)
            VU_adj = sp.coo_matrix((np.ones(self.VU_edges.shape[0]), (self.VU_edges[:, 0], self.VU_edges[:, 1])),
                                   shape=(self.opt["num_item"], self.opt["num_user"]),
                                   dtype=np.float32)
            all_adj = sp.coo_matrix((np.ones(self.all_edges.shape[0]), (self.all_edges[:, 0], self.all_edges[:, 1])), shape=(
                self.opt["num_item"] + self.opt["num_user"], self.opt["num_item"] + self.opt["num_user"]), dtype=np.float32)
        self.UV_adj = normalize(UV_adj)
        self.VU_adj = normalize(VU_adj)
        if self_loop == 0:
            self.all_adj = normalize(all_adj)
        else:
            self.all_adj = normalize(all_adj + sp.eye(all_adj.shape[0]))
        self.UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj) 
        self.VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
        self.all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)