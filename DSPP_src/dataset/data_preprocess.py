from utils.LoadGraph import *
from utils.BipartiteGraph import BipartiteGraph
import pickle
import os
import argparse

class T_json(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, opt, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence, item_sequence_id, item_timediffs_sequence,timestamp_sequence, begin_index, end_index, evaluation = False, sliding_window = None, sliding_index=None, sliding_timestamp = None, total_graph=None, last_Bipartite = None):

        self.opt = opt
        self.eval = evaluation

        self.user_sequence_id = user_sequence_id[begin_index:end_index]
        self.user_timediffs_sequence=user_timediffs_sequence[begin_index:end_index]
        self.user_previous_itemid_sequence = user_previous_itemid_sequence[begin_index:end_index]
        self.item_sequence_id = item_sequence_id[begin_index:end_index]
        self.item_timediffs_sequence = item_timediffs_sequence[begin_index:end_index]
        self.timestamp_sequence = timestamp_sequence[begin_index:end_index]
        self.subbatch_count = 0
        self.total_graph = BipartiteGraph(self.opt)
        if evaluation:
            self.sliding_window = copy.deepcopy(sliding_window)
            self.sliding_index = copy.deepcopy(sliding_index)
            self.sliding_timestamp = copy.deepcopy(sliding_timestamp)
            self.last_Bipartite = copy.deepcopy(last_Bipartite)
            self.total_graph = copy.deepcopy(total_graph)

        else :
            self.sliding_window = {}
            self.sliding_index = {}
            self.sliding_timestamp = {}
            for u_id in range(self.opt["num_user"]):
                self.sliding_window[u_id] = [self.opt["num_item"] - 1] * self.opt["attention_window"]
                self.sliding_timestamp[u_id] = [0] * self.opt["attention_window"]
                self.sliding_index[u_id] = 0


        self.preprocess()
        # chunk into batches
        print("{} batches created!".format(len(self.T_batch)))

    def preprocess(self):
        # SET BATCHING TIMESPAN AND CREATE T-BATCH DATASET
        self.T_batch = []
        if self.eval:
            self.T_batch_graph = [copy.deepcopy(self.last_Bipartite)]
        else :
            self.T_batch_graph = [BipartiteGraph(self.opt), BipartiteGraph(self.opt)] # one more zero graph
            self.T_batch.append([])
        user_batch_level = defaultdict(int)
        item_batch_level = defaultdict(int)
        cur = self.timestamp_sequence[0]
        cur_index = 0

        while True:
            if cur_index == len(self.user_sequence_id):
                break
            if self.T_batch_graph[-1].graph_interaction_number:
                # self.T_batch_graph[-1].change_to_tensor()
                T_subgraph = BipartiteGraph(self.opt)
                if self.opt["append_graph"]:
                    if cur_index:
                        T_subgraph.add_pre_graph(self.T_batch_graph[-1].bigraph, self.opt["decay"])
                    else :
                        T_subgraph.add_pre_graph(self.T_batch_graph[-1].bigraph, 1.0) # begin the test
                    self.T_batch_graph.append(T_subgraph)
                else :
                    self.T_batch_graph.append(T_subgraph)
                self.T_batch.append([])
                user_batch_level = defaultdict(int)
                item_batch_level = defaultdict(int)


            cur += self.opt["T_batch_timespan"]
            while (cur_index < len(self.user_sequence_id)) and (self.timestamp_sequence[cur_index] < cur):
                cur_user = self.user_sequence_id[cur_index]
                cur_item = self.item_sequence_id[cur_index]
                cur_time = self.timestamp_sequence[cur_index]

                # if self.eval: # cold start user
                #     if cur_user not in self.total_graph.bigraph:
                #         cur_index += 1
                #         continue

                self.T_batch_graph[-1].add_edge(cur_user, cur_item)
                self.total_graph.add_edge(cur_user, cur_item)

                cur_level = max(user_batch_level[cur_user], item_batch_level[cur_item])

                if cur_level == len(self.T_batch[-1]):
                    self.T_batch[-1].append([])
                self.T_batch[-1][cur_level].append(
                    [cur_user, cur_item, cur_time])
                user_batch_level[cur_user] = cur_level + 1
                item_batch_level[cur_item] = cur_level + 1

                cur_index += 1

        self.T_batch_tensor = []
        for T_id, one_t_batch in enumerate(self.T_batch):

            batches = []

            # merging batches for fast train-speed, note that is validate the time constraints in most cases.
            # deleting the following codes will keep the original t-batch algorithm in JODIE
            one_batch = []
            for level in one_t_batch:
                one_batch += level

            one_t_batch = [one_batch]

            for level in one_t_batch:
                cur_user = []
                cur_item = []
                cur_pred_time = []
                cur_timestamp_window = []
                cur_attention_window = []

                for interaction in level:
                    if self.sliding_index[interaction[0]] > 5:
                        # now
                        cur_user.append(interaction[0])
                        cur_item.append(interaction[1])
                        #given the current interaction time and previous interacted items, predicting the target item
                        if self.opt["network"] == "lastfm": # the time gap of one minus
                            minute = 3174
                        else :
                            minute = 60

                        cur_index = self.sliding_index[interaction[0]] % self.opt["attention_window"]
                        sliding_window = self.sliding_timestamp[interaction[0]][cur_index:] + self.sliding_timestamp[interaction[0]][:cur_index]
                        maxtime = max(sliding_window)
                        now_time_stamp = []
                        for time in sliding_window:
                            now_time_stamp.append(min(maxtime - time, 24 * 60 * minute) / minute)
                        cur_timestamp_window.append(now_time_stamp)
                        cur_attention_window.append(self.sliding_window[interaction[0]][cur_index:] + self.sliding_window[interaction[0]][:cur_index])
                        pred_time = []
                        for id, time in enumerate(sliding_window):
                            if id:
                                pred_time.append(min(sliding_window[id] - sliding_window[id-1], 24 * 60 * minute) / minute)
                        pred_time.append((interaction[2] - maxtime) / minute)
                        cur_pred_time.append(pred_time)
                    # update
                    self.add_interaction(interaction[0], interaction[1], interaction[2])


                cur_user = [cur_user[i:i + self.opt["batch_size"]] for i in range(0, len(cur_user), self.opt["batch_size"])]
                cur_item = [cur_item[i:i + self.opt["batch_size"]] for i in
                                range(0, len(cur_item), self.opt["batch_size"])]
                cur_pred_time = [cur_pred_time[i:i + self.opt["batch_size"]] for i in
                                 range(0, len(cur_pred_time), self.opt["batch_size"])]
                cur_timestamp_window = [cur_timestamp_window[i:i + self.opt["batch_size"]] for i in
                                 range(0, len(cur_timestamp_window), self.opt["batch_size"])]
                cur_attention_window = [cur_attention_window[i:i + self.opt["batch_size"]] for i in
                                         range(0, len(cur_attention_window), self.opt["batch_size"])]


                self.subbatch_count += len(cur_user)

                for split_id in range(len(cur_user)):
                    batch_manner = {}
                    batch_manner["user"] = cur_user[split_id]
                    batch_manner["item"] = cur_item[split_id]
                    batch_manner["pred_time"] = cur_pred_time[split_id]
                    batch_manner["timestamp_window"] = cur_timestamp_window[split_id]
                    batch_manner["attention_window"] = cur_attention_window[split_id]

                    pos_itemid = []
                    neg_itemid = []

                    user = cur_user[split_id]
                    item = cur_item[split_id]
                    pos_attention_window = cur_attention_window[split_id]

                    for cur_id, cur_user_id in enumerate(user):
                        pos = pos_attention_window[cur_id][1:]
                        pos.append(item[cur_id])
                        pos_itemid.append(pos)
                        # neg_itemid.append(
                        #     self.total_graph.random_list_multi_neg(cur_user_id, pos, self.opt["negative_sample"]))
                        #
                        # tmp = self.T_batch_graph[T_id].random_pos(cur_user_id)
                        # long_pos_itemid.append(tmp)
                        # tmp = self.T_batch_graph[T_id].random_neg(cur_user_id)
                        # long_neg_itemid.append(tmp)
                        # neg_itemid.append(self.T_batch_graph[T_id].random_list_multi_neg(cur_user_id, pos, self.opt["negative_sample"]))
                        neg_itemid.append(self.T_batch_graph[T_id].random_list_multi_neg_with_some_pos(cur_user_id, pos, self.opt["negative_sample"]))

                    batch_manner["pos_itemid"] = pos_itemid
                    batch_manner["neg_itemid"] = neg_itemid
                    # batch_manner["long_pos_itemid"] = long_pos_itemid
                    # batch_manner["long_neg_itemid"] = long_neg_itemid
                    batches.append(batch_manner)
            self.T_batch_tensor.append(batches)

    def add_interaction(self, user, item, timestamp):
        self.sliding_window[user][self.sliding_index[user] % self.opt["attention_window"]] = item
        self.sliding_timestamp[user][self.sliding_index[user] % self.opt["attention_window"]] = timestamp
        self.sliding_index[user] += 1
        # self.sliding_index[user] %= self.opt["attention_window"]

    def __len__(self):
        return len(self.T_batch)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.T_batch):
            raise IndexError
        if self.T_batch_graph[key].graph_interaction_number==0:
            self.T_batch_graph[key]=None
        return (self.T_batch_tensor[key], self.T_batch_graph[key]) # the previous graph

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

def process(data_dir, network, snapshots, sequence_length):
    # LOAD DATA
    opt = {}
    opt["data_dir"] = data_dir
    opt["network"] = network
    opt["T_batch_number"] = snapshots
    opt["undebug"] = 1
    opt["batch_size"] = 1024 #larger batch size for fast training speed, but 128 for better result
    opt["train_proportion"] = 0.8 # train / test rate
    opt["attention_window"] = sequence_length
    opt["negative_sample"] = 10
    opt["append_graph"] = True # the bounded graph ([start, end]) or the total graph ([0, end])
    opt["decay"] = 0.7 # decay rate to build the dynamic graph snapshots
    opt["weight"] = True
    if "la" in network:
        opt["pos_neg"] = 3
    else:
        opt["pos_neg"] = 1

    opt["datapath"] = "{}/{}.csv".format(data_dir, network)
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
     item2id, item_sequence_id, item_timediffs_sequence,
     timestamp_sequence] = load_network(opt)


    # dataset statistics
    num_interactions = len(user_sequence_id)
    num_user = len(user2id)
    num_item = len(item2id) + 1 # the pading item

    # set the time gap
    timespan = timestamp_sequence[-1] - timestamp_sequence[0]  # overall time
    opt["T_batch_timespan"] = timespan / opt["T_batch_number"]  # the number of graph snapshot

    opt["num_user"] = num_user
    opt["num_item"] = num_item
    print("***\n Network statistics:\n  %d users\n  %d items\n  %d interactions\n\n" % (
    num_user, num_item, num_interactions))

    train_end_idx = int(num_interactions * opt["train_proportion"])
    validation_end_idx = int(num_interactions * (opt["train_proportion"] + 0.1))
    test_end_idx = int(num_interactions * (opt["train_proportion"] + 0.2))

    train_batch = T_json(opt, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
                             item_sequence_id, item_timediffs_sequence, timestamp_sequence, 0,
                             train_end_idx)
    test_batch = T_json(opt, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
                           item_sequence_id, item_timediffs_sequence, timestamp_sequence,
                           train_end_idx, test_end_idx, evaluation=True, sliding_window=train_batch.sliding_window,
                           sliding_index=train_batch.sliding_index, sliding_timestamp=train_batch.sliding_timestamp, total_graph=train_batch.total_graph,
                           last_Bipartite=train_batch.T_batch_graph[-1])

    print(len(train_batch.T_batch))
    print(len(train_batch.T_batch_graph))

    print(len(test_batch.T_batch))
    print(len(test_batch.T_batch_graph))

    ensure_dir("{}/{}".format(opt["data_dir"], opt["network"]+"_"+str(opt["T_batch_number"])+"_"+str(opt["attention_window"])))
    print("{}/{}/{}.pkl".format(opt["data_dir"], opt["network"]+"_"+str(opt["T_batch_number"])+"_"+str(opt["attention_window"]),"opt"))
    opt["subbatch_count"] = train_batch.subbatch_count
    pickle.dump(opt, open("{}/{}/{}.pkl".format(opt["data_dir"], opt["network"]+"_"+str(opt["T_batch_number"])+"_"+str(opt["attention_window"]),"opt"), "wb"))
    pickle.dump(test_batch.total_graph, open("{}/{}/{}.pkl".format(opt["data_dir"], opt["network"]+"_"+str(opt["T_batch_number"])+"_"+str(opt["attention_window"]), "total_graph"), "wb"))
    for seq_id, batch in enumerate(train_batch):
        pickle.dump(batch, open("{}/{}/{}/{}.pkl".format(opt["data_dir"],opt["network"]+"_"+str(opt["T_batch_number"])+"_"+str(opt["attention_window"]), "train", seq_id), "wb"))
    for seq_id, batch in enumerate(test_batch):
        pickle.dump(batch, open("{}/{}/{}/{}.pkl".format(opt["data_dir"],opt["network"]+"_"+str(opt["T_batch_number"])+"_"+str(opt["attention_window"]), "test", seq_id), "wb"))



def process_debug(data_dir, network):
    # LOAD DATA
    opt = {}
    opt["data_dir"] = data_dir
    opt["network"] = network
    opt["T_batch_number"] = 20
    opt["undebug"] = 0
    opt["batch_size"] = 2048
    opt["train_proportion"] = 0.8
    opt["attention_window"] = 20
    opt["negative_sample"] = 10
    opt["append_graph"] = True
    opt["decay"] = 0.7
    opt["weight"] = True
    if "la" in network:
        opt["pos_neg"] = 3
    else:
        opt["pos_neg"] = 1

    opt["datapath"] = "{}/{}.csv".format(data_dir, network)
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
     item2id, item_sequence_id, item_timediffs_sequence,
     timestamp_sequence] = load_network(opt)


    print(max(item_sequence_id))
    # dataset statistics
    num_interactions = len(user_sequence_id)
    num_user = len(user2id)
    num_item = len(item2id) + 1 # the padding item

    # set the time gap
    timespan = timestamp_sequence[-1] - timestamp_sequence[0]  # total time
    opt["T_batch_timespan"] = timespan / opt["T_batch_number"]  # the time interval

    opt["num_user"] = num_user
    opt["num_item"] = num_item

    print(opt)

    print("***\n Network statistics:\n  %d users\n  %d items\n  %d interactions\n\n" % (
    num_user, num_item, num_interactions))

    train_end_idx = int(num_interactions * opt["train_proportion"])
    validation_end_idx = int(num_interactions * (opt["train_proportion"] + 0.1))
    test_end_idx = int(num_interactions * (opt["train_proportion"] + 0.2))

    train_batch = T_json(opt, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
                             item_sequence_id, item_timediffs_sequence, timestamp_sequence, 0,
                             train_end_idx)
    test_batch = T_json(opt, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
                           item_sequence_id, item_timediffs_sequence, timestamp_sequence,
                           train_end_idx, test_end_idx, evaluation=True, sliding_window=train_batch.sliding_window,
                           sliding_index=train_batch.sliding_index, sliding_timestamp=train_batch.sliding_timestamp, total_graph=train_batch.total_graph,
                           last_Bipartite=train_batch.T_batch_graph[-1])
    print(len(train_batch.T_batch))
    print(len(train_batch.T_batch_graph))

    print(len(test_batch.T_batch))
    print(len(test_batch.T_batch_graph))

    opt["subbatch_count"] = train_batch.subbatch_count
    ensure_dir("{}/{}".format(opt["data_dir"], opt["network"]+"debug"))
    print("{}/{}/{}.pkl".format(opt["data_dir"], opt["network"]+"debug","opt"))
    pickle.dump(opt, open("{}/{}/{}.pkl".format(opt["data_dir"], opt["network"]+"debug","opt"), "wb"))
    pickle.dump(test_batch.total_graph, open("{}/{}/{}.pkl".format(opt["data_dir"], opt["network"]+"debug", "total_graph"), "wb"))
    for seq_id, batch in enumerate(train_batch):
        pickle.dump(batch, open("{}/{}/{}/{}.pkl".format(opt["data_dir"],opt["network"]+"debug", "train", seq_id), "wb"))
    for seq_id, batch in enumerate(test_batch):
        pickle.dump(batch, open("{}/{}/{}/{}.pkl".format(opt["data_dir"],opt["network"]+"debug", "test", seq_id), "wb"))

def ensure_dir(d):
    if not os.path.exists(d):
        print("Directory {} do not exist; creating...".format(d))
        os.makedirs(d)
        os.makedirs(d+"/train")
        os.makedirs(d+"/test")




parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='.')
parser.add_argument('--network', type=str, default='reddit')
parser.add_argument('--graphsnapshot', type=int, default=1024)
parser.add_argument('--sequence_length', type=int, default=80)

args = parser.parse_args()
opt = vars(args)

print(opt)

process(opt["data_dir"], opt["network"], opt["graphsnapshot"], opt["sequence_length"])
# process_debug(opt["data_dir"], opt["network"])

# python data_preprocess.py --network reddit --graphsnapshot 1024 --sequence_length 60
