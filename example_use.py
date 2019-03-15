import torch
from torch import nn
import torch.nn.functional as F
import random
#### Load pretrained bert model
from bert import BERTLM
from data import Vocab, CLS, SEP, MASK
def init_bert_model(args, device, bert_vocab):
    bert_ckpt= torch.load(args.bert_path, map_location='cpu')
    bert_args = bert_ckpt['args']
    bert_vocab = Vocab(bert_vocab, min_occur_cnt=bert_args.min_occur_cnt, specials=[CLS, SEP, MASK])
    bert_model = BERTLM(device, bert_vocab, bert_args.embed_dim, bert_args.ff_embed_dim, bert_args.num_heads, bert_args.dropout, bert_args.layers, bert_args.approx)
    bert_model.load_state_dict(bert_ckpt['model'])
    bert_model = bert_model.cuda(device)
    return bert_model, bert_vocab, bert_args
#####
"""
The above function loads all information from a pretrained BERT, including
the model, the vocabulary, and the hyper-parameters.
Now you should live on your own.
Below gives an example.
"""
#####
### Define your own model
def ListsToTensor(xs, vocab):

    batch_size = len(xs)
    lens = [ len(x)+2 for x in xs]
    mx_len = max(lens)
    ys = []
    for i, x in enumerate(xs):
        y =  vocab.token2idx([CLS]+x+[SEP]) + ([vocab.padding_idx]*(mx_len - lens[i]))
        ys.append(y)

    data = torch.LongTensor(ys).t_().contiguous()
    return data

def batchify(data, vocab):
    src = ListsToTensor([x[0] for x in data], vocab)
    tgt = ListsToTensor([x[1] for x in data], vocab)
    return src, tgt

class myDataLoader(object):
    def __init__(self, filename, vocab, batch_size, for_train):
        all_data = [ [ x.split() for x in line.strip().split('|') ] for line in open(filename, encoding='utf8').readlines()]

        self.data = []
        for d in all_data:
            skip = not (len(d) == 2)
            for j, i in enumerate(d):
                if not for_train:
                    d[j] = i[:30]
                    if len(d[j]) == 0:
                        d[j] = [UNK]
                if len(i) ==0 or len(i) > 30:
                    skip = True
            if not (skip and for_train):
                self.data.append(d)

        self.batch_size = batch_size 
        self.vocab =vocab
        self.train = for_train
    
    def __iter__(self):
        idx = list(range(len(self.data)))
        if self.train:
            random.shuffle(idx)
        cur = 0
        while cur < len(idx):
            data = [self.data[i] for i in idx[cur:cur+self.batch_size]]
            cur += self.batch_size
            yield batchify(data, self.vocab)
        return

def label_smoothed_nll_loss(log_probs, target, eps):
    #log_probs: N x C
    #target: N
    nll_loss =  -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    if eps == 0.:
        return nll_loss
    smooth_loss = -log_probs.sum(dim=-1)
    eps_i = eps / log_probs.size(-1)
    loss = (1. - eps) * nll_loss + eps_i * smooth_loss
    return loss

class myModel(nn.Module):
    def __init__(self, bert_model, embed_dim, dropout, device):
        super(myModel, self).__init__()
        self.bert_model = bert_model
        self.scorer = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.device = device
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.scorer.weight, std=0.02)
        nn.init.constant_(self.scorer.bias, 0.)
    
    def forward(self, src, tgt, fine_tune=False):
        bsz = src.size(1)
        src = src.cuda(self.bert_model.device)
        tgt = tgt.cuda(self.bert_model.device)
        
        x = self.bert_model.work(src)[1].cuda(self.device)
        y = self.bert_model.work(tgt)[1].cuda(self.device)
        if not fine_tune:
            x = x.detach()
            y = y.detach()
        x = F.dropout(x, p=self.dropout, training=self.training)
        y = F.dropout(y, p=self.dropout, training=self.training)
        scores = torch.mm(self.scorer(x), y.transpose(0, 1)) # bsz x bsz
        log_probs = F.log_softmax(scores, -1)

        gold = torch.arange(bsz).cuda(self.device)

        _, pred = torch.max(log_probs, -1)

        acc = torch.sum(torch.eq(gold, pred).float()) / bsz
        #print pred

        loss = label_smoothed_nll_loss(log_probs, gold, 0.1)
        #F.nll_loss(log_probs, torch.arange(bsz).cuda(), reduction = "elementwise_mean")
        loss = loss.mean()
        return loss, acc

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', type=str)
    parser.add_argument('--bert_vocab', type=str)
    parser.add_argument('--train_data',type=str)
    parser.add_argument('--dev_data',type=str)
    parser.add_argument('--train_batch_size',type=int)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--dropout',type=float)

    parser.add_argument('--print_every', type=int)
    parser.add_argument('--eval_every', type=int)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--fine_tune', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_config()
    bert_model, bert_vocab, bert_args = init_bert_model(args, args.gpu_id, args.bert_vocab)

    train_data = myDataLoader(args.train_data, bert_vocab, args.train_batch_size, for_train=True)
    dev_data = myDataLoader(args.dev_data, bert_vocab, args.train_batch_size, for_train=True)

    print ('data is ready!')
    model = myModel(bert_model, bert_args.embed_dim, args.dropout, args.gpu_id)
    model = model.cuda(args.gpu_id)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    loss_accumulated = 0.
    acc_accumulated = 0.
    batches_processed = 0
    best_dev_acc = 0
    while True:
        for src_input, tgt_input in train_data:
            optimizer.zero_grad()
            loss, acc = model(src_input, tgt_input, args.fine_tune)

            loss_accumulated += loss.item()
            acc_accumulated += acc
            batches_processed += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if batches_processed % args.print_every == -1 % args.print_every:
                print ("Batch %d, loss %.5f, acc %.5f"%(batches_processed, loss_accumulated / batches_processed, acc_accumulated / batches_processed))
            if batches_processed % args.eval_every == -1 % args.eval_every:
                model.eval()
                dev_acc = 0.
                dev_batches = 0
                for src_input, tgt_input in dev_data:
                    _, acc = model(src_input, tgt_input)
                    dev_acc += acc
                    dev_batches += 1
                dev_acc = dev_acc / dev_batches
                if best_dev_acc < dev_acc:
                    best_dev_acc = dev_acc
                    torch.save({'args':args, 'model':model.state_dict()}, 'ckpt/batch%d_acc_%.3f'%(batches_processed, dev_acc))

                print ("Dev Batch %d, acc %.5f"%(batches_processed, dev_acc))
                model.train()
