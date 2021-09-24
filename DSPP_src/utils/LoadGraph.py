from __future__ import division
import numpy as np
import random
import sys
import operator
import copy
from collections import defaultdict
import os, re
import pickle
import argparse
from sklearn.preprocessing import scale

# LOAD THE NETWORK
def load_network(args, time_scaling=False):
    '''
    This function loads the input network.

    The network should be in the following format:
    One line per interaction/edge.
    Each line should be: user, item, timestamp, state label, array of features.
    Timestamp should be in cardinal format (not in datetime).
    State label should be 1 whenever the user state changes, 0 otherwise. If there are no state labels, use 0 for all interactions.
    Feature list can be as long as desired. It should be atleast 1 dimensional. If there are no features, use 0 for all interactions.
    '''

    # fw = open("jodiewikipedia.txt","w",encoding="utf-8")
    network = args["network"]
    datapath = args["datapath"]

    user_sequence = []
    item_sequence = []
    feature_sequence = []
    timestamp_sequence = []
    start_timestamp = None
    y_true_labels = []
    if args["debug"]:
        ok = 1
    else :
        ok = 0
    print("\n\n**** Loading %s network from file: %s ****" % (network, datapath))
    f = open(datapath,"r")
    f.readline()
    for cnt, l in enumerate(f):
        # FORMAT: user, item, timestamp, state label, feature list
        if ok:
            if cnt == 10000:
                break
        ls = l.strip().split(",")
        user_sequence.append(ls[0])
        item_sequence.append(ls[1])
        if start_timestamp is None:
            start_timestamp = float(ls[2])
        # fw.write(ls[0])
        # fw.write("\t")
        # fw.write(ls[1])
        # fw.write("\t")
        # fw.write(ls[2])
        # fw.write("\n")
        timestamp_sequence.append(float(ls[2]) - start_timestamp)
        y_true_labels.append(int(ls[3])) # label = 1 at state change, 0 otherwise
        feature_sequence.append(list(map(float, ls[4:])))
    f.close()
    # fw.close()

    user_sequence = np.array(user_sequence)
    item_sequence = np.array(item_sequence)
    timestamp_sequence = np.array(timestamp_sequence)

    print("Formating item sequence")
    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)

    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    print("Formating user sequence")
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]

    """
    user_next_itemid_sequence = []
    user_next_itemid_timestamp = []
    item_next_userid_sequence = []
    item_next_userid_timestamp = []

    user_prev_index = defaultdict(lambda: -1)
    item_prev_index = defaultdict(lambda: -1)

    for cnt, user in enumerate(user_sequence):
        cur_user = user_sequence_id[cnt]
        cur_item = item_sequence_id[cnt]
        cur_time = timestamp_sequence[cnt]

        user_next_itemid_sequence.append(num_items)
        item_next_userid_sequence.append(num_users)

        if user_prev_index[cur_user] != -1:
            user_next_itemid_sequence[user_prev_index[cur_user]] = cur_item
            user_next_itemid_timestamp[user_prev_index[cur_user]] = cur_time
        if item_prev_index[cur_item] != -1:
            item_next_userid_sequence[item_prev_index[cur_item]] = cur_user
            item_next_userid_timestamp[item_prev_index[cur_item]] = cur_time

        user_prev_index[cur_user] = cnt
        item_prev_index[cur_item] = cnt
    """

    if time_scaling:
        print("Scaling timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)

    print("*** Network loading completed ***\n\n")
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, \
        item2id, item_sequence_id, item_timedifference_sequence, \
        timestamp_sequence, \
        feature_sequence, \
        y_true_labels]