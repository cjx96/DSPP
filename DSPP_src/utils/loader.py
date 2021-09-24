"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
from utils.BipartiteGraph import BipartiteGraph
from collections import defaultdict
import copy
import pickle
import os
import time
from torch.utils.data import Dataset, DataLoader

class preprocess_DataLoader(object):
    def __init__(self, opt, evaluation = False):
        self.opt = opt
        self.total_graph = pickle.load(open("{}/{}/{}.pkl".format(opt["data_dir"], opt["network"], "total_graph"), "rb"))
        if evaluation:
            self.train_or_test = "test"
        else:
            self.train_or_test = "train"
        self.data_size = int(len(os.listdir("{}/{}/{}/".format(opt["data_dir"], opt["network"], self.train_or_test))))

        self.graph_cache = {}
        self.batches_cache = {}
    def __len__(self):
        return self.data_size

    def __getitem__(self, key):
        """ Get a batch with index. """
        if key not in self.graph_cache:
            (batches, graph) = pickle.load(
                open("{}/{}/{}/{}.pkl".format(self.opt["data_dir"], self.opt["network"], self.train_or_test, key),
                     "rb"))
            self.graph_cache[key] = graph
            if self.graph_cache[key] is not None:
                self.graph_cache[key].change_to_tensor()
            if self.opt["cuda"]:
                for i in range(len(batches)):
                    batches[i]["user"] = torch.LongTensor(batches[i]["user"]).cuda()
                    batches[i]["item"] = torch.LongTensor(batches[i]["item"]).cuda()
                    batches[i]["pred_time"] = torch.FloatTensor(batches[i]["pred_time"]).cuda()
                    batches[i]["attention_window"] = torch.LongTensor(batches[i]["attention_window"]).cuda()
                    batches[i]["timestamp_window"] = torch.FloatTensor(batches[i]["timestamp_window"]).cuda()
                    batches[i]["pos_itemid"] = torch.LongTensor(batches[i]["pos_itemid"]).cuda()
                    batches[i]["neg_itemid"] = torch.LongTensor(batches[i]["neg_itemid"]).cuda()
                    # batches[i]["long_pos_itemid"] = torch.LongTensor(batches[i]["long_pos_itemid"]).cuda()
                    # batches[i]["long_neg_itemid"] = torch.LongTensor(batches[i]["long_neg_itemid"]).cuda()
            else :
                for i in range(len(batches)):
                    batches[i]["user"] = torch.LongTensor(batches[i]["user"])
                    batches[i]["item"] = torch.LongTensor(batches[i]["item"])
                    batches[i]["pred_time"] = torch.FloatTensor(batches[i]["pred_time"])
                    batches[i]["attention_window"] = torch.LongTensor(batches[i]["attention_window"])
                    batches[i]["timestamp_window"] = torch.FloatTensor(batches[i]["timestamp_window"])
                    batches[i]["pos_itemid"] = torch.LongTensor(batches[i]["pos_itemid"])
                    batches[i]["neg_itemid"] = torch.LongTensor(batches[i]["neg_itemid"])
                    # batches[i]["long_pos_itemid"] = torch.LongTensor(batches[i]["long_pos_itemid"])
                    # batches[i]["long_neg_itemid"] = torch.LongTensor(batches[i]["long_neg_itemid"])
            self.batches_cache[key] = batches
        return (self.batches_cache[key], self.graph_cache[key])
    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


class preprocess_graph_DataLoader(object):
    def __init__(self, opt, evaluation = False):
        self.opt = opt
        self.total_graph = pickle.load(open("{}/{}/{}.pkl".format(opt["data_dir"], opt["network"], "total_graph"), "rb"))
        if evaluation:
            self.train_or_test = "test"
        else:
            self.train_or_test = "train"
        self.data_size = int(len(os.listdir("{}/{}/{}/".format(opt["data_dir"], opt["network"], self.train_or_test))))
        self.graph_cache = {}
    def __len__(self):
        return self.data_size

    def __getitem__(self, key):
        """ Get a batch with index. """
        (batches, graph) = pickle.load(open("{}/{}/{}/{}.pkl".format(self.opt["data_dir"], self.opt["network"], self.train_or_test, key), "rb"))

        if key not in self.graph_cache:
            graph_cache[key] = graph
            if graph_cache[key] is not None:
                graph_cache[key].change_to_tensor()
        for i in range(len(batches)):
            batches[i][0] = torch.LongTensor(batches[i][0]) # cur_user
            batches[i][1] = torch.FloatTensor(batches[i][1]) # cur_user_timediff
            batches[i][2] = torch.LongTensor(batches[i][2]) # cur_user_prev_itemid
            batches[i][3] = torch.LongTensor(batches[i][3]) # cur_item
            batches[i][4] = torch.FloatTensor(batches[i][4]) # cur_item_timediff
            batches[i][5] = torch.FloatTensor(batches[i][5]) # cur_feature_window
            batches[i][6] = torch.LongTensor(batches[i][6]) # cur_y_state_label
            batches[i][7] = torch.LongTensor(batches[i][7]) # cur_attention_window
            batches[i][8] = torch.FloatTensor(batches[i][8]) # cur_timestamp_window
            batches[i][9] = torch.FloatTensor(batches[i][9])  # cur_feature
        return (batches, graph_cache[key])


    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

class ATTENTION_DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, opt, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence, item_sequence_id, item_timediffs_sequence,timestamp_sequence, feature_sequence, y_true, begin_index, end_index, evaluation = False, sliding_window = None, sliding_index=None, sliding_timestamp = None, sliding_feature = None, total_graph=None, last_Bipartite = None):

        self.opt = opt
        self.eval = evaluation

        self.user_sequence_id = user_sequence_id[begin_index:end_index]
        self.user_timediffs_sequence=user_timediffs_sequence[begin_index:end_index]
        self.user_previous_itemid_sequence = user_previous_itemid_sequence[begin_index:end_index]
        self.item_sequence_id = item_sequence_id[begin_index:end_index]
        self.item_timediffs_sequence = item_timediffs_sequence[begin_index:end_index]
        self.timestamp_sequence = timestamp_sequence[begin_index:end_index]
        self.feature_sequence = feature_sequence[begin_index:end_index]
        self.y_true = y_true[begin_index:end_index]
        self.subbatch_count = 0
        self.total_graph = BipartiteGraph(self.opt)
        if evaluation:
            self.sliding_window = copy.deepcopy(sliding_window)
            self.sliding_index = copy.deepcopy(sliding_index)
            self.sliding_timestamp = copy.deepcopy(sliding_timestamp)
            self.sliding_feature = copy.deepcopy(sliding_feature)
            self.last_Bipartite = copy.deepcopy(last_Bipartite)
            self.total_graph = copy.deepcopy(total_graph)

        else :
            self.sliding_window = {}
            self.sliding_index = {}
            self.sliding_timestamp = {}
            self.sliding_feature = {}
            for u_id in range(self.opt["num_user"]):
                self.sliding_window[u_id] = [self.opt["num_item"] - 1] * self.opt["attention_window"]
                self.sliding_timestamp[u_id] = [0] * self.opt["attention_window"]
                self.sliding_feature[u_id] = [[0] * self.opt["interaction_features"]] * self.opt["attention_window"]
                self.sliding_index[u_id] = 0


        if evaluation:
            self.preprocess()
        else :
            self.preprocess()
        # chunk into batches
        print("{} batches created!".format(len(self.T_batch)))

    def preprocess_for_predict(self):
        # SET BATCHING TIMESPAN AND CREATE T-BATCH DATASET
        pass

    def preprocess(self):
        # SET BATCHING TIMESPAN AND CREATE T-BATCH DATASET
        self.T_batch = [[]]
        if self.eval:
            self.T_batch_graph = [copy.deepcopy(self.last_Bipartite)]
        else :
            self.T_batch_graph = [BipartiteGraph(self.opt)]
        user_batch_level = defaultdict(int)
        item_batch_level = defaultdict(int)
        cur = self.timestamp_sequence[0]
        cur_index = 0

        while True:

            if self.T_batch_graph[-1].graph_interaction_number:
                self.T_batch_graph[-1].change_to_tensor()
                if cur_index == len(self.user_sequence_id):
                    break
                self.T_batch.append([])

                if self.opt["append_graph"]:
                    self.T_batch_graph.append(copy.deepcopy(self.T_batch_graph[-1]))
                else :
                    self.T_batch_graph.append(BipartiteGraph(self.opt))
                user_batch_level = defaultdict(int)
                item_batch_level = defaultdict(int)
            if cur_index == len(self.user_sequence_id):
                break

            cur += self.opt["T_batch_timespan"]
            while (cur_index < len(self.user_sequence_id)) and (self.timestamp_sequence[cur_index] < cur):
                cur_user = self.user_sequence_id[cur_index]
                cur_user_timediff = self.user_timediffs_sequence[cur_index]
                cur_user_prev_itemid = self.user_previous_itemid_sequence[cur_index]
                cur_item = self.item_sequence_id[cur_index]
                cur_item_timediff = self.item_timediffs_sequence[cur_index]
                cur_y_state_label = self.y_true[cur_index]
                cur_feature = self.feature_sequence[cur_index]
                cur_time = self.timestamp_sequence[cur_index]

                if self.eval: # cold start user
                    if cur_user not in self.total_graph.bigraph:
                        cur_index += 1
                        continue

                self.T_batch_graph[-1].add_edge(cur_user, cur_item)
                self.total_graph.add_edge(cur_user, cur_item)

                cur_level = max(user_batch_level[cur_user], item_batch_level[cur_item])
                if cur_level == len(self.T_batch[-1]):
                    self.T_batch[-1].append([])
                self.T_batch[-1][cur_level].append(
                    [cur_user, cur_user_timediff, cur_user_prev_itemid, cur_item, cur_item_timediff,
                     cur_y_state_label,cur_feature,cur_time])
                user_batch_level[cur_user] = cur_level + 1
                item_batch_level[cur_item] = cur_level + 1

                cur_index += 1

        self.T_batch_tensor = []
        for one_t_batch in self.T_batch:

            batches = []


            for level in one_t_batch:

                cur_user = []
                cur_user_timediff = []
                cur_user_prev_itemid = []
                cur_item = []
                cur_item_timediff = []
                cur_y_state_label = []
                cur_timestamp_window = []
                cur_attention_window = []
                cur_feature_window = []
                cur_feature = []

                for interaction in level:

                    # now
                    cur_user.append(interaction[0])
                    cur_user_timediff.append(interaction[1])
                    cur_user_prev_itemid.append(interaction[2])
                    cur_item.append(interaction[3])
                    cur_item_timediff.append(interaction[4])
                    cur_y_state_label.append(interaction[5])
                    cur_feature.append(interaction[6])

                    # previous
                    now_time_stamp = []
                    for time in self.sliding_timestamp[interaction[0]]:
                        now_time_stamp.append(min(interaction[7] - time, 3000)/100)
                    cur_timestamp_window.append(now_time_stamp[self.sliding_index[interaction[0]]:] + now_time_stamp[:self.sliding_index[interaction[0]]])
                    cur_attention_window.append(self.sliding_window[interaction[0]][self.sliding_index[interaction[0]]:] + self.sliding_window[interaction[0]][:self.sliding_index[interaction[0]]])
                    cur_feature_window.append(self.sliding_feature[interaction[0]][self.sliding_index[interaction[0]]:] + self.sliding_feature[interaction[0]][:self.sliding_index[interaction[0]]])
                    self.add_item(interaction[0], interaction[3],interaction[6],interaction[7])



                cur_user = [cur_user[i:i + self.opt["batch_size"]] for i in range(0, len(cur_user), self.opt["batch_size"])]
                cur_user_timediff = [cur_user_timediff[i:i + self.opt["batch_size"]] for i in
                                range(0, len(cur_user_timediff), self.opt["batch_size"])]
                cur_user_prev_itemid = [cur_user_prev_itemid[i:i + self.opt["batch_size"]] for i in
                                range(0, len(cur_user_prev_itemid), self.opt["batch_size"])]
                cur_item = [cur_item[i:i + self.opt["batch_size"]] for i in
                                range(0, len(cur_item), self.opt["batch_size"])]
                cur_item_timediff = [cur_item_timediff[i:i + self.opt["batch_size"]] for i in
                                range(0, len(cur_item_timediff), self.opt["batch_size"])]
                cur_y_state_label = [cur_y_state_label[i:i + self.opt["batch_size"]] for i in
                                     range(0, len(cur_y_state_label), self.opt["batch_size"])]
                cur_timestamp_window = [cur_timestamp_window[i:i + self.opt["batch_size"]] for i in
                                 range(0, len(cur_timestamp_window), self.opt["batch_size"])]
                cur_attention_window = [cur_attention_window[i:i + self.opt["batch_size"]] for i in
                                         range(0, len(cur_attention_window), self.opt["batch_size"])]
                cur_feature_window = [cur_feature_window[i:i + self.opt["batch_size"]] for i in
                               range(0, len(cur_feature_window), self.opt["batch_size"])]
                cur_feature = [cur_feature[i:i + self.opt["batch_size"]] for i in
                                      range(0, len(cur_feature), self.opt["batch_size"])]

                self.subbatch_count += len(cur_user)

                for split_id in range(len(cur_user)):
                    batches.append(
                        [cur_user[split_id], cur_user_timediff[split_id], cur_user_prev_itemid[split_id],
                            cur_item[split_id], cur_item_timediff[split_id], cur_feature_window[split_id],
                            cur_y_state_label[split_id], cur_attention_window[split_id], cur_timestamp_window[split_id], cur_feature[split_id]])

                # for split_id in range(len(cur_user)):
                #     if len(cur_user[split_id]) > 1:
                #         if len(batches):
                #             if len(batches[-1])==1:
                #                 for cur_id in range(len(cur_user[split_id])):
                #                     batches[-1][0].append(cur_user[split_id][cur_id])
                #                     batches[-1][1].append(cur_user_timediff[split_id][cur_id])
                #                     batches[-1][2].append(cur_user_prev_itemid[split_id][cur_id])
                #                     batches[-1][3].append(cur_item[split_id][cur_id])
                #                     batches[-1][4].append(cur_item_timediff[split_id][cur_id])
                #                     batches[-1][5].append(cur_feature[split_id][cur_id])
                #                     batches[-1][6].append(cur_y_state_label[split_id][cur_id])
                #                     batches[-1][7].append(cur_attention_window[split_id][cur_id])
                #                     batches[-1][8].append(cur_timestamp[split_id][cur_id])
                #             else :
                #                 batches.append(
                #                     [cur_user[split_id], cur_user_timediff[split_id], cur_user_prev_itemid[split_id],
                #                      cur_item[split_id], cur_item_timediff[split_id], cur_feature[split_id],
                #                      cur_y_state_label[split_id], cur_attention_window[split_id], cur_timestamp[split_id]])
                #         else :
                #             batches.append([cur_user[split_id],cur_user_timediff[split_id],cur_user_prev_itemid[split_id],cur_item[split_id],cur_item_timediff[split_id],cur_feature[split_id],cur_y_state_label[split_id],cur_attention_window[split_id],cur_timestamp[split_id]])
                #     elif len(batches):
                #         for cur_id in range(len(cur_user[split_id])):
                #             batches[-1][0].append(cur_user[split_id][cur_id])
                #             batches[-1][1].append(cur_user_timediff[split_id][cur_id])
                #             batches[-1][2].append(cur_user_prev_itemid[split_id][cur_id])
                #             batches[-1][3].append(cur_item[split_id][cur_id])
                #             batches[-1][4].append(cur_item_timediff[split_id][cur_id])
                #             batches[-1][5].append(cur_feature[split_id][cur_id])
                #             batches[-1][6].append(cur_y_state_label[split_id][cur_id])
                #             batches[-1][7].append(cur_attention_window[split_id][cur_id])
                #             batches[-1][8].append(cur_timestamp[split_id][cur_id])
                #     else:
                #         batches.append([cur_user[split_id], cur_user_timediff[split_id], cur_user_prev_itemid[split_id],
                #                         cur_item[split_id], cur_item_timediff[split_id], cur_feature[split_id],
                #                         cur_y_state_label[split_id], cur_attention_window[split_id],
                #                         cur_timestamp[split_id]])

            for i in range(len(batches)):
                batches[i][0] = torch.LongTensor(batches[i][0]) # cur_user
                batches[i][1] = torch.FloatTensor(batches[i][1]) # cur_user_timediff
                batches[i][2] = torch.LongTensor(batches[i][2]) # cur_user_prev_itemid
                batches[i][3] = torch.LongTensor(batches[i][3]) # cur_item
                batches[i][4] = torch.FloatTensor(batches[i][4]) # cur_item_timediff
                batches[i][5] = torch.FloatTensor(batches[i][5]) # cur_feature_window
                batches[i][6] = torch.LongTensor(batches[i][6]) # cur_y_state_label
                batches[i][7] = torch.LongTensor(batches[i][7]) # cur_attention_window
                batches[i][8] = torch.FloatTensor(batches[i][8]) # cur_timestamp_window
                batches[i][9] = torch.FloatTensor(batches[i][9])  # cur_feature
            self.T_batch_tensor.append(batches)

    def add_item(self,user,item,feature,timestamp):
        self.sliding_window[user][self.sliding_index[user]] = item
        self.sliding_feature[user][self.sliding_index[user]] = feature
        self.sliding_timestamp[user][self.sliding_index[user]] = timestamp
        self.sliding_index[user] += 1
        self.sliding_index[user] %= self.opt["attention_window"]

    def __len__(self):
        return len(self.T_batch)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.T_batch):
            raise IndexError

        if key:
            graph = self.T_batch_graph[key - 1]
        else :
            if self.eval:
                graph = self.last_Bipartite
            else:
                graph = None

        return (self.T_batch_tensor[key], graph)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)




class RNN_DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, opt, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence, item_sequence_id, item_timediffs_sequence,timestamp_sequence, feature_sequence, y_true, begin_index, end_index, evaluation = False):

        self.opt = opt
        self.eval = evaluation

        self.user_sequence_id = user_sequence_id[begin_index:end_index]
        self.user_timediffs_sequence=user_timediffs_sequence[begin_index:end_index]
        self.user_previous_itemid_sequence = user_previous_itemid_sequence[begin_index:end_index]
        self.item_sequence_id = item_sequence_id[begin_index:end_index]
        self.item_timediffs_sequence = item_timediffs_sequence[begin_index:end_index]
        self.timestamp_sequence = timestamp_sequence[begin_index:end_index]
        self.feature_sequence = feature_sequence[begin_index:end_index]
        self.y_true = y_true[begin_index:end_index]

        if not evaluation:
            self.preprocess()
        else :
            self.preprocess()
        # chunk into batches
        print("{} batches created!".format(len(self.T_batch)))

    def preprocess_for_predict(self):
        # SET BATCHING TIMESPAN AND CREATE T-BATCH DATASET
        pass

    def preprocess(self):
        # SET BATCHING TIMESPAN AND CREATE T-BATCH DATASET
        self.T_batch = [[]]
        self.T_batch_graph = [BipartiteGraph(self.opt)]
        user_batch_level = defaultdict(int)
        item_batch_level = defaultdict(int)
        T_batch_size = []
        cur = self.timestamp_sequence[0] 
        cur_index = 0

        while True:

            if self.T_batch_graph[-1].graph_interaction_number:
                self.T_batch_graph[-1].change_to_tensor()
                if cur_index == len(self.user_sequence_id):
                    break
                self.T_batch.append([])
                self.T_batch_graph.append(BipartiteGraph(self.opt))
                user_batch_level = defaultdict(int)
                item_batch_level = defaultdict(int)
            if cur_index == len(self.user_sequence_id):
                break

            cur += self.opt["T_batch_timespan"]
            while (cur_index < len(self.user_sequence_id)) and (self.timestamp_sequence[cur_index] < cur):
                cur_user = self.user_sequence_id[cur_index]
                cur_user_timediff = self.user_timediffs_sequence[cur_index]
                cur_user_prev_itemid = self.user_previous_itemid_sequence[cur_index]
                cur_item = self.item_sequence_id[cur_index]
                cur_item_timediff = self.item_timediffs_sequence[cur_index]
                cur_feature = self.feature_sequence[cur_index]
                cur_y_state_label = self.y_true[cur_index]

                self.T_batch_graph[-1].add_edge(cur_user, cur_item)

                cur_level = max(user_batch_level[cur_user], item_batch_level[cur_item])
                if cur_level == len(self.T_batch[-1]):
                    self.T_batch[-1].append([])
                self.T_batch[-1][cur_level].append(
                    [cur_user, cur_user_timediff, cur_user_prev_itemid, cur_item, cur_item_timediff,
                     cur_y_state_label,cur_feature])
                user_batch_level[cur_user] = cur_level + 1
                item_batch_level[cur_item] = cur_level + 1

                cur_index += 1

        self.T_batch_tensor = []
        for one_t_batch in self.T_batch:

            batches = []

            for level in one_t_batch:
                T_batch_size.append(len(level))
                cur_user = []
                cur_user_timediff = []
                cur_user_prev_itemid = []
                cur_item = []
                cur_item_timediff = []
                cur_y_state_label = []
                cur_feature = []

                for interaction in level:

                    cur_user.append(interaction[0])
                    cur_user_timediff.append(interaction[1])
                    cur_user_prev_itemid.append(interaction[2])
                    cur_item.append(interaction[3])
                    cur_item_timediff.append(interaction[4])
                    cur_y_state_label.append(interaction[5])
                    cur_feature.append(interaction[6])

                cur_user = torch.LongTensor(cur_user)
                cur_user_timediff = torch.FloatTensor(cur_user_timediff)
                cur_user_prev_itemid = torch.LongTensor(cur_user_prev_itemid)
                cur_item = torch.LongTensor(cur_item)
                cur_item_timediff = torch.FloatTensor(cur_item_timediff)
                cur_y_state_label = torch.LongTensor(cur_y_state_label)
                cur_feature = torch.FloatTensor(cur_feature)

                batches.append(
                    [cur_user, cur_user_timediff, cur_user_prev_itemid, cur_item, cur_item_timediff, cur_y_state_label,
                     cur_feature])
            self.T_batch_tensor.append(batches)
        # sorted(T_batch_size)
        print(len(T_batch_size))
        print(max(T_batch_size))
        input()

    def __len__(self):
        return len(self.T_batch)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.T_batch):
            raise IndexError

        tensor_batches = self.T_batch_tensor[key]
        if key:
            graph = self.T_batch_graph[key - 1]
        else :
            graph = None

        return (tensor_batches, graph)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

