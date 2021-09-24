"""
__author__ = "caojiangxia"
"""

import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.loader import preprocess_DataLoader as DataLoader
from utils.BipartiteGraph import BipartiteGraph
from utils.LoadGraph import *
from utils import torch_utils, helper
from utils.scorer import *
import torch.nn.functional as F
from model.trainer import DSPPTrainer
# import fitlog # optional package


# torch.cuda.set_device(2)

parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--data_dir', type=str, default='dataset')
parser.add_argument('--network', type=str, default='lastfm')

# model part
parser.add_argument('--GNN', type=int, default=2, help="The layer of TAL.")
parser.add_argument('--attention_layer', type=int, default=8, help="The layer of attention encoder.")
parser.add_argument('--feature_dim', type=int, default=128, help='Initialize node embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=128, help='GNN network hidden embedding dimension.')
parser.add_argument('--dropout', type=float, default=0.3, help='Normal dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.97, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=30, help='Decay learning rate after this epoch.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--negative_sample', type=int, default=10)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--lambda', type=float, default=0.3)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--attention_window', type=int, default=50)
parser.add_argument('--attention_heads', type=int, default=1)
parser.add_argument('--position_dim', type=int, default=128)
parser.add_argument('--biased_position_dim', type=int, default=128)
parser.add_argument('--metric', type=str, default="dot")
parser.add_argument('--test_update', action='store_false', default=True)
parser.add_argument('--undebug', action='store_true', default=False)

# train part
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--log_step', type=int, default=2000, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--seed', type=int, default=2040)
parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--save_node_feature', action='store_true', default=False, help='save node feature')

def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

args = parser.parse_args()
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()
# make opt
opt = vars(args)
seed_everything(opt["seed"])

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")


if opt["undebug"]:
    pass
else:
    opt["cuda"] = False
    opt["cpu"] = True
    opt["batch_size"] = 5
    opt["feature_dim"] = 8
    opt["hidden_dim"] = 8
    opt["position_dim"] = 8
    opt["biased_position_dim"] = 8
    opt["attention_window"] = 7
    opt['log_step'] = 2000
    fitlog.debug()
    # opt["cuda"] = False
    # opt["cpu"] = True

# fitlog.commit(__file__)
# fitlog.set_log_dir("logs/")          #  the save dir

tmp_opt = pickle.load(open("{}/{}/{}.pkl".format(opt["data_dir"], opt["network"], "opt"), "rb"))
for k in tmp_opt.keys():
    if k == "debug":
        continue
    if k == "undebug":
        continue
    if k == "network":
        continue
    if k == "data_dir":
        continue
    opt[k] = tmp_opt[k]


opt["original_dim"] = opt["feature_dim"]
opt["position_dim"] = opt["position_dim"]
opt["biased_position_dim"] = opt["biased_position_dim"]
opt["long_dim"] = opt["original_dim"]
opt["short_dim"] = opt["original_dim"]
opt["final_dim"] = opt["original_dim"]

# fitlog.add_hyper(copy.deepcopy(opt))
# print model info
helper.print_config(opt)

# setting the dataloader
train_batch = DataLoader(opt)
dev_batch = DataLoader(opt, True)

# model
if not opt['load']:
    trainer = DSPPTrainer(opt)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = DSPPTrainer(opt)
    trainer.load(model_file)

valider = DSPPTrainer(opt)
dev_score_history = [0]
current_lr = opt['lr']
global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/epoch), lr: {:.6f}'

max_steps = opt["subbatch_count"] * opt['num_epoch']

print(opt["subbatch_count"], " per epoch")
# start training
for epoch in range(0, opt['num_epoch'] + 1):
    trainer.init_embedding()
    epoch_start_time = time.time()
    train_loss = 0
    for seq_id, batch in enumerate(train_batch):
        if epoch == 0:
            # Loading data to GPU
            continue

        tensor_batches = batch[0]
        prev_graph = batch[1]
        begin = time.time()

        for sub_batch_id, sub_batch in enumerate(tensor_batches):
            global_step += 1
            loss = trainer.forward_batch(sub_batch, prev_graph)
            train_loss += loss

        trainer.update_graph_prev(prev_graph)
        # print(f"batch time : {time.time() - begin}, time: {seq_id}")

    if epoch == 0:
        print("cuda done!")
        continue

    train_loss = train_loss / opt['subbatch_count'] # avg loss per batch
    print("pred_loss: ", trainer.pred_loss / opt['subbatch_count'])
    print("mse_loss: ", trainer.mse_loss / opt['subbatch_count'])
    duration = time.time() - epoch_start_time
    print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                            opt['num_epoch'], train_loss, duration, current_lr))
    # fitlog.add_loss(value=train_loss, name="{}-Loss".format(opt['network']), step=global_step, epoch=epoch)

    if epoch % 5:
        continue
    # eval model
    print("Evaluating on dev set...")
    trainer.save(str("saved_models/")+str(opt["id"])+".pt")
    valider.load(str("saved_models/")+str(opt["id"])+".pt")
    valider.change_optimizer(current_lr)
    valider.to_cuda()

    all_rank = []
    val_step = 0
    for seq_id, batch in enumerate(dev_batch):
        tensor_batches = batch[0]
        prev_graph = batch[1]
        for sub_batch in tensor_batches:
            val_step+=1
            rank = valider.forward_batch(sub_batch, prev_graph, evaluate=True)
            all_rank += rank
            if opt["test_update"]:
                valider.forward_batch(sub_batch, prev_graph)
        valider.update_graph_prev(prev_graph)
    print("val_pred_loss: ", trainer.pred_loss / val_step)
    print("val_mse_loss: ", trainer.mse_loss / val_step)

    val_all_rank = all_rank[:-len(all_rank) // 2]
    val_mrr = np.mean([1.0 / r for r in val_all_rank])
    val_rec10 = sum(np.array(val_all_rank) <= 10) * 1.0 / len(val_all_rank)
    print(datetime.now(), " epoch {}: VAL_MRR = {:.6f}, VAL_RECALL10 = {:.6f}".format(
        epoch, val_mrr, val_rec10))

    test_all_rank = all_rank[-len(all_rank)//2:]
    test_mrr = np.mean([1.0 / r for r in test_all_rank])
    test_rec10 = sum(np.array(test_all_rank) <= 10) * 1.0 / len(test_all_rank)
    print(datetime.now(), " epoch {}: TEST_MRR = {:.6f}, TEST_RECALL10 = {:.6f}".format(
            epoch, test_mrr, test_rec10))

    # fitlog.add_metric(value=val_mrr, name="{}-val_mrr".format(opt['network']), step=global_step, epoch=epoch)
    # fitlog.add_metric(value=val_rec10, name="{}-val_rec10".format(opt['network']), step=global_step, epoch=epoch)
    # fitlog.add_metric(value=test_mrr, name="{}-test_mrr".format(opt['network']), step=global_step, epoch=epoch)
    # fitlog.add_metric(value=test_rec10, name="{}-test_rec10".format(opt['network']), step=global_step, epoch=epoch)

    dev_score = test_mrr
    dev_rec10 = test_rec10

    # save
    if epoch == 1 or dev_score > max(dev_score_history):
        print("new best model.")
        # fitlog.add_best_metric(name='{}-mrr'.format(opt['network']), value=dev_score)
        # fitlog.add_best_metric(name='{}-rec10'.format(opt['network']), value=dev_rec10)

    # lr schedule
    if epoch > opt['decay_epoch'] and dev_score <= dev_score_history[-1]:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]

    file_logger.log("{}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_score, max(dev_score_history)))
    # fitlog.add_to_line("{}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_score, max(dev_score_history)))

print("Training ended with {} epochs.".format(epoch))
# fitlog.finish()

# CUDA_VISIBLE_DEVICES=0 python -u train.py --id debug --network lastfmdebug --undebug