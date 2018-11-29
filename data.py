import random
import torch
import numpy as np

PAD, UNK, CLS, SEP, MASK = '<-PAD->', '<-UNK->', '<-CLS->', '<-SEP->', '<-MASK->'

def ListsToTensor(xs, vocab=None):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        if vocab is not None:
            y = vocab.token2idx(x) + [vocab.padding_idx]*(max_len -len(x))
        else:
            y = x + [0]*(max_len -len(x))
        ys.append(y)
    data = torch.LongTensor(ys).t_().contiguous()
    return data

def random_mask(x, vocab):
    masked_x, mask = [], []
    _mask = np.random.choice(4, len(x), p = [0.85, 0.15*0.8, 0.15*0.1, 0.15*0.1])
    for mi, xi in zip(_mask, x):
        if mi == 0:
            masked_x.append(xi)
            mask.append(1)
        else:
            if mi == 1:
                masked_x.append(MASK)
            elif mi == 2:
                masked_x.append(vocab.random_token())
            elif mi == 3:
                masked_x.append(xi)
            mask.append(0)
    return masked_x, mask

def batchify(data, vocab):
    truth, inp, seg, msk = [], [], [], []
    nxt_snt_flag = []
    for a, b, r in data:
        x = ['CLS']+a+['SEP']+b+['SEP']
        truth.append(x)
        seg.append([0]*(len(a)+2) + [1]*(len(b)+1))
        masked_x, mask = random_mask(x, vocab)
        inp.append(masked_x)
        msk.append(mask)
        nxt_snt_flag.append(1)
        
        x = ['CLS']+a+['SEP']+r+['SEP']
        truth.append(x)
        seg.append([0]*(len(a)+2) + [1]*(len(b)+1))
        masked_x, mask = random_mask(x, vocab)
        inp.append(masked_x)
        msk.append(mask)
        nxt_snt_flag.append(0)
    
    truth = ListsToTensor(truth, vocab)
    inp = ListsToTensor(inp, vocab)
    seg = ListsToTensor(seg)
    msk = ListsToTensor(msk).to(torch.uint8)
    nxt_snt_flag = torch.ByteTensor(nxt_snt_flag)

    return truth, inp, seg, msk, nxt_snt_flag


class DataLoader(object):
    def __init__(self, vocab, filename, batch_size, max_len):
        self.data = []
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_len = max_len
        for line in open(filename).readlines():
            d = line.strip().split()[:max_len]
            self.data.append(d)

    def __iter__(self):
        idx = list(range(len(self.data)))
        random.shuffle(idx)
        batches = []
        for a, r in enumerate(idx[:-1]):
            b = a + 1
            if max( len(self.data[b]) , len(self.data[r]) ) + len(self.data[a]) <= self.max_len:
                batches.append((self.data[a], self.data[b], self.data[r]))
        idx = 0
        while idx < len(batches):
            yield batchify(batches[idx:idx+self.batch_size], self.vocab)
            idx += self.batch_size

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        idx2token = [PAD, UNK] + ( specials if specials is not None else [])
        for line in open(filename).readlines():
            token, cnt = line.strip().split()
            if int(cnt) >= min_occur_cnt:
                idx2token.append(token)
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    @property
    def size(self):
        return len(self._idx2token)
    
    @property
    def unk_idx(self):
        return self._unk_idx
    
    @property
    def padding_idx(self):
        return self._padding_idx
    
    def random_token(self):
        return self.idx2token(np.random.randint(self.size))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)
