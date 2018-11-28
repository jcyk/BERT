import numpy as np

with open('vocab', 'w') as fo:
    for i in range(5):
        fo.write(str(i)+'\t50\n')

with open('train', 'w') as fo:
    for i in range(10000):
        line = ' '.join([str(x) for x in np.random.randint(5, size=5)])
        fo.write(line+'\n')