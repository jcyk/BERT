import numpy as np
import random
with open('vocab', 'w') as fo:
    for i in range(20):
        fo.write(str(i)+'\t50\n')

all_ = list(range(20))
with open('train', 'w') as fo:
    for i in range(100):
        st = random.randint(1, 10)
        line = ' '.join([str(x) for x in all_[st:st+8]])
        fo.write(line+'\n')

