#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from bert import BERTLM
from data import Vocab, DataLoader

import argparse, os

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--ff_embed_dim', type=int)
    parser.add_argument('--num_heads', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument('--dropout', type=float)

    parser.add_argument('--train_data', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)

    parser.add_argument('--world_size', type=int)
    parser.add_argument('--gpus', type=int)
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    parser.add_argument('--start_rank', type=int)

    return parser.parse_args()

def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size

def run(args, local_rank):
    """ Distributed Synchronous """
    torch.manual_seed(1234)
    vocab = Vocab(args.vocab, min_occur_cnt=5)
    model = BERTLM(local_rank, vocab, args.embed_dim, args.ff_embed_dim, args.num_heads, args.dropout, args.layers)
    model = model.cuda(local_rank)
    
    torch.manual_seed(1234+dist.get_rank())
    optimizer = optim.Adam(model.parameters(),1e-4, (0.9, 0.999), weight_decay=0.01)

    train_data = DataLoader(vocab, args.train_data, args.batch_size, args.max_len)

    batch_acm = 0
    acc_acm, ntokens_acm, acc_nxt_acm, npairs_acm = 0., 0., 0., 0.
    loss_acm = 0.
    while True:
        for truth, inp, seg, msk, nxt_snt_flag in train_data:

            truth = truth.cuda(local_rank)
            inp = inp.cuda(local_rank)
            seg = seg.cuda(local_rank)
            msk = msk.cuda(local_rank)
            nxt_snt_flag = nxt_snt_flag.cuda(local_rank)


            optimizer.zero_grad()
            loss, acc, ntokens, acc_nxt, npairs = model(truth, inp, seg, msk, nxt_snt_flag)
            loss_acm += loss.item()
            acc_acm += acc
            ntokens_acm += ntokens
            acc_nxt_acm += acc_nxt
            npairs_acm += npairs
            batch_acm +=1

            loss.backward()
            average_gradients(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if batch_acm%args.print_every == -1%args.print_every:
                print ('batch_acm %d, acc %.3f, nxt_acc %.3f'%(batch_acm, acc_acm/ntokens_acm, acc_nxt_acm/npairs_acm))
                acc_acm, ntokens_acm, acc_nxt_acm, npairs_acm = 0., 0., 0., 0.
                loss_acm = 0.
            if batch_acm%args.save_every == -1%args.save_every:
                torch.save({'args':args, 'model':model.state_dict()}, 'ckpt/batch_%d_rank_%d'%(batch_acm, dist.get_rank()))

def init_processes(args, local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT
    dist.init_process_group(backend, rank=args.start_rank+local_rank, world_size=args.world_size)
    fn(args, local_rank)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_config()

    processes = []
    for rank in range(args.gpus):
        p = mp.Process(target=init_processes, args=(args, rank, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
