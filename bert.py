import torch
from torch import nn
import torch.nn.functional as F

from utils import gelu
from transformer import TransformerLayer, Embedding, LearnedPositionalEmbedding

class BERTLM(nn.Module):
    def __init__(self, local_rank, vocab, embed_dim, ff_embed_dim, num_heads, dropout, layers):
        super(BERTLM, self).__init__()
        self.vocab = vocab
        self.tok_embed = Embedding(self.vocab.size, embed_dim, self.vocab.padding_idx)
        self.pos_embed = LearnedPositionalEmbedding(embed_dim, device=local_rank)
        self.seg_embed = nn.Embedding(2, embed_dim)

        self.out_proj_bias = nn.Parameter(torch.Tensor(self.vocab.size))

        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(TransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout))

        self.one_more = nn.Linear(embed_dim, embed_dim)
        self.one_more_norm = nn.LayerNorm(embed_dim)
        self.nxt_snt_pred = nn.Linear(embed_dim, 1)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.out_proj_bias, 0.)
        nn.init.constant_(self.nxt_snt_pred.bias, 0.)
        nn.init.constant_(self.seg_embed.weight, 0.)
        nn.init.xavier_uniform_(self.nxt_snt_pred.weight)
        nn.init.xavier_uniform_(self.one_more.weight)
        nn.init.constant_(self.one_more.bias, 0.)

    def forward(self, truth, inp, seg, msk, nxt_snt_flag):
        seq_len, bsz = inp.size()
        x = self.tok_embed(inp) + self.seg_embed(seg) + self.pos_embed(inp)
        x = F.dropout(x, p=self.dropout, training= self.training)
        for layer in self.layers:
            x, _ ,_ = layer(x)

        y = self.one_more_norm(gelu(self.one_more(x)))
        out_proj_weight = self.tok_embed.weight
        log_probs = torch.log_softmax(F.linear(y, out_proj_weight, self.out_proj_bias), -1)

        _, pred = log_probs.max(-1)
        acc = torch.eq(pred, truth).float().masked_select(msk).sum().item()
        tot_tokens = msk.float().sum().item()
        
        loss = F.nll_loss(log_probs.view(seq_len*bsz, -1), truth.view(-1), reduction='none').view(seq_len, bsz)
        loss = loss.masked_select(msk)

        z = x[0]
        nxt_snt_pred = torch.sigmoid(self.nxt_snt_pred(z).squeeze(1))
        nxt_snt_acc = torch.eq(torch.gt(nxt_snt_pred, 0.5), nxt_snt_flag).float().sum().item()
        nxt_snt_loss = F.binary_cross_entropy(nxt_snt_pred, nxt_snt_flag.float(), reduction='none')

        
        tot_loss = loss.sum() / tot_tokens + nxt_snt_loss.sum() / bsz
        pred = pred.masked_select(msk)
        return pred, tot_loss, acc, tot_tokens, nxt_snt_acc, bsz
