import sys
import glob
import torch

files = glob.glob(sys.argv[1] + '.*.pt')
for f in files:
    cur_dataset = torch.load(f)
    print("============================================")
    print(f)
    print('number of examples: %d' % len(cur_dataset))