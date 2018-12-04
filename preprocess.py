
from multiprocessing import Pool
from collections import Counter
import sys, re
import argparse

from google_bert import BasicTokenizer

BUFSIZE = 2048000

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str)
    parser.add_argument('--tgt_file', type=str)
    parser.add_argument('--nprocessors', type=int)
    return parser.parse_args()


_split_set = set(['！', '？', '。'])
def _is_split_point(ch):
    if ch in _split_set:
        return True
    return False

def work_wiki_char(line):
    "This function only works for zhwiki at char level"
    tokenizer = BasicTokenizer() 
    line = line.strip()
    if line == "":
        return []
    if line.startswith("</doc>"):
        return [[]]
    if line.startswith("<doc id="):
        return [[]]
    char_seq = tokenizer.tokenize(line)
    res = []
    sent = []
    for ch in char_seq:
        sent.append(ch)
        if len(sent)>=20 and _is_split_point(ch):
            res.append(sent)
            sent = []
    if sent:
        if len(sent) <= 3 and len(res)>0:
            res[-1].extend(sent)
        else:
            res.append(sent)
    return res

if __name__ == "__main__":
    args = parse_config()
    pool = Pool(args.nprocessors)
    stream = open(args.src_file, encoding='utf8')
    cnt = Counter()
    with open(args.tgt_file, 'w', encoding ='utf8') as fo:
        while True:
            lines = stream.readlines(BUFSIZE)
            if not lines:
                break
            res = pool.map(work_wiki_char, lines, len(lines)//args.nprocessors)
            if not res:
                continue
            all_lines = []
            for lines in res:
                all_lines.extend(lines)
            empty = True
            for line in all_lines:
                if line:
                    cnt.update(line)
                    fo.write(' '.join(line)+'\n')
                    empty = False
                else: 
                    if not empty:
                        fo.write('\n')
                    empty = True
    with open(args.tgt_file+'_vocab', 'w', encoding ='utf8') as fo:
        for x, y in cnt.most_common():
            fo.write(x+'\t'+str(y)+'\n')
