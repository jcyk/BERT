import random
import torch
import numpy as np

from google_bert import create_instances_from_document

PAD, UNK, CLS, SEP, MASK = '<-PAD->', '<-UNK->', '<-CLS->', '<-SEP->', '<-MASK->'
BUFSIZE = 2048000

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

def random_mask(tokens, masked_lm_prob, max_predictions_per_seq, vocab):
    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    masked_tokens, mask = [], []
    cand = []
    for i, token in enumerate(tokens):
        if token == CLS or token == SEP:
            continue
        cand.append(i)
    random.shuffle(cand)
    cand = set(cand[:num_to_predict])

    masked_tokens, mask = [], []
    for i, token in enumerate(tokens):
        if i in cand:
            if random.random() < 0.8:
                masked_tokens.append(MASK)
            else:
                if random.random() < 0.5:
                    masked_tokens.append(token)
                else:
                    masked_tokens.append(vocab.random_token())
            mask.append(1)
        else:
            masked_tokens.append(token)
            mask.append(0)
    return masked_tokens, mask

def _back_to_text_for_check(x, vocab):
    w = x.t().tolist()
    for sent in vocab.idx2token(w):
        print (' '.join(sent))
    
def batchify(data, vocab):
    truth, inp, seg, msk = [], [], [], []
    nxt_snt_flag = []
    for a, b, r in data:
        x = [CLS]+a+[SEP]+b+[SEP]
        truth.append(x)
        seg.append([0]*(len(a)+2) + [1]*(len(b)+1))
        masked_x, mask = random_mask(x, 0.15, 20, vocab)
        inp.append(masked_x)
        msk.append(mask)
        if r:
            nxt_snt_flag.append(0)
        else:
            nxt_snt_flag.append(1)

    truth = ListsToTensor(truth, vocab)
    inp = ListsToTensor(inp, vocab)
    seg = ListsToTensor(seg)
    msk = ListsToTensor(msk).to(torch.uint8)
    nxt_snt_flag = torch.ByteTensor(nxt_snt_flag)
    return truth, inp, seg, msk, nxt_snt_flag

class DataLoader(object):
    def __init__(self, vocab, filename, batch_size, max_len):
        self.batch_size = batch_size
        self.vocab = vocab
        self.max_len = max_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')

    def __iter__(self):
        
        lines = self.stream.readlines(BUFSIZE)

        if not lines:
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)

        docs = [[]]
        for line in lines:
            tokens = line.strip().split()
            if tokens:
                docs[-1].append(tokens)
            else:
                docs.append([])
        docs = [x for x in docs if x]
        random.shuffle(docs)

        data = []
        for idx, doc in enumerate(docs):
            data.extend(create_instances_from_document(docs, idx, self.max_len))

        idx = 0
        while idx < len(data):
            yield batchify(data[idx:idx+self.batch_size], self.vocab)
            idx += self.batch_size

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        idx2token = [PAD, UNK] + ( specials if specials is not None else [])
        for line in open(filename, encoding='utf8').readlines():
            try: 
                token, cnt = line.strip().split()
            except:
                continue
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
