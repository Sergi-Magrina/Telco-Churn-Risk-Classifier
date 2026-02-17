import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:

    # global random seeds for reproducibility across numpy, random, and torch.

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

