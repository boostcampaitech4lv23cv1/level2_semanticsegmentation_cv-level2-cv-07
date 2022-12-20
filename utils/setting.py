import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random

def set_seed(n):# seed 고정
    random_seed = n
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

if __name__=="__main__":
    set_seed(21)