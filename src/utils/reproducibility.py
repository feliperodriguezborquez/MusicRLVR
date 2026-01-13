import torch
import random
import numpy as np
import os

def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers on all GPUs and CPU.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")
