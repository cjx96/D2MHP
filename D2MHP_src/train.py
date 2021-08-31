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
from model.trainer import PPTrainer
from utils import torch_utils, helper
from utils.scorer import *
from utils.loader import *
import json
import codecs
import tqdm

# torch.cuda.set_device(1)

parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--data_dir', type=str, default='ml-1m')

# model part
parser.add_argument('--model', type=str, default="d2mhp",
                    help='model name.[sahp, rmtpp, irnn, nhp, appvae, vepp, d2mhp and so on]')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--maxlen', type=int, default=200)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--have_event', type=bool, default=True)
parser.add_argument('--no_event', type=bool, default=True)
parser.add_argument('--rmtpp_loss', type=bool, default=True)
parser.add_argument('--time_scale', type=float, default=0.01)
parser.add_argument('--kl_decay', type=float, default=0.1)
parser.add_argument('--variant', action='store_true', default=False)
parser.add_argument('--disentangle', action='store_true', default=False)
parser.add_argument('--x_dim', type=int, default=64)
parser.add_argument('--h_dim', type=int, default=64)
parser.add_argument('--z_dim', type=int, default=64)
parser.add_argument('--dis_k', type=int, default=3)
parser.add_argument('--identity_dim', type=int, default=32)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--mi_loss', type=float, default=1)

# train part
parser.add_argument('--num_epoch', type=int, default=1000, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--seed', type=int, default=2040)
parser.add_argument('--load', dest='load', action='store_true', default=False, help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')
parser.add_argument('--undebug', action='store_false', default=True)


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

if opt["data_dir"] == "ml-1m":
    opt["data_dir"] = "src/data/ml-1m.txt"
elif opt["data_dir"] == "taobao_item":
    opt["data_dir"] = "src/data/taobao_item2000.txt"
elif opt["data_dir"] == "so":
    opt["data_dir"] = "src/data/so.txt"
elif opt["data_dir"] == "retweet":
    opt["data_dir"] = "src/data/retweet.txt"

seed_everything(opt["seed"])
model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)



print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
# dataset = data_partition(opt['data_dir'])
dataset = data_partition(opt['data_dir'], opt)
[entity_train, entity_valid, entity_test, entitynum, eventnum, timenum] = dataset
num_batch = len(entity_train) // opt["batch_size"]


opt["eventnum"] = eventnum

# model
if not opt['load']:
    trainer = PPTrainer(opt)
    sampler = WarpSampler_point(entity_train, entitynum, eventnum, batch_size=opt["batch_size"], maxlen=opt["maxlen"],
                                    n_workers=3)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = PPTrainer(opt)
    trainer.load(model_file)

global_step = 0
current_lr = opt["lr"]
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} , lr: {:.6f}'
max_steps = opt['num_epoch'] * num_batch

begin_time = time.time()
dev_score_history = [0]
# start training
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    for _ in range(num_batch):
        global_step += 1
        u, seq, time_seq, pos, neg, target_time = sampler.next_batch()  # tuples to ndarray
        target_time = list(target_time)
        alpha_mask = []  # batch_seq_event
        for batch_id in range(len(time_seq)):  # batch
            seq_alpha_mask = [0] * (opt["eventnum"] + 1)
            for id in range(0, len(time_seq[batch_id]) - 1):
                seq_alpha_mask[seq[batch_id][id]] = (time_seq[batch_id][-1] - time_seq[batch_id][id]) / time_seq[batch_id][-1]
            alpha_mask.append(seq_alpha_mask)
            target_time[batch_id] = (time_seq[batch_id][-1] - time_seq[batch_id][-2]) / 10
        alpha_mask = np.array(alpha_mask)
        alpha_mask = torch.FloatTensor(alpha_mask)
        if args.cuda:
            alpha_mask = alpha_mask.cuda()
        u, seq, time_seq, pos, neg, target_time = np.array(u), np.array(seq), np.array(time_seq), np.array(pos), np.array(neg), np.array(target_time)
        if args.cuda:
            u = torch.LongTensor(u).cuda()
            seq = torch.LongTensor(seq).cuda()
            time_seq = torch.FloatTensor(time_seq).cuda()
            pos = torch.LongTensor(pos).cuda()
            neg = torch.LongTensor(neg).cuda()
            target_time = torch.FloatTensor(target_time).cuda()
        else:
            u = torch.LongTensor(u)
            seq = torch.LongTensor(seq)
            time_seq = torch.FloatTensor(time_seq)
            pos = torch.LongTensor(pos)
            neg = torch.LongTensor(neg)
            target_time = torch.FloatTensor(target_time)
        loss = trainer.train_batch(seq, time_seq, pos, neg, alpha_mask, target_time)
        train_loss += loss

    print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                            opt['num_epoch'], train_loss / num_batch, current_lr))
    if epoch % 20:
        continue
        # pass

    # eval model
    print("Evaluating on dev set...")
    trainer.model.eval()
    t_test = evaluate(trainer, dataset, args)
    t_valid = evaluate_valid(trainer, dataset, args)
    dev_score_history.append(t_valid[0])

"""
CUDA_VISIBLE_DEVICES=0 python -u train.py --model d2mhp --undebug --data_dir ml-1m --dis_k 2
"""