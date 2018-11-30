import numpy as np

with open('vocab', 'w') as fo:
    for i in range(5):
        fo.write(str(i)+'\t50\n')

with open('train', 'w') as fo:
    for i in range(1000):
        line = ' '.join([str(x%2) for x in range(20)])
        line = ' '.join([str(x%3) for x in range(20)])
        fo.write(line+'\n')

