import torch, os, sys
from itertools import combinations
from dataclasses import dataclass, field
from typing import List, Tuple

from time import perf_counter
from contextlib import contextmanager
import numpy as np
import pandas as pd
import random
from scipy.stats import kruskal, chi2_contingency, fisher_exact, ttest_ind
from sklearn.model_selection import GroupShuffleSplit

SEED = 42

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

def printl(nlines=1):
    print('\n'*nlines,'='*80,'\n')


def set_all_seeds(seed=42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"All seeds fixed to {seed}.")

def worker_init_fn(worker_id):
    """Reseed each DataLoader worker."""
    worker_seed = 42 + worker_id
    random.seed(worker_seed)
    import numpy as np
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BINARY_TASKS = [
    "diabetic_retinopathy", "macular_edema", "scar", "nevus", "amd",
    "vascular_occlusion", "hypertensive_retinopathy", "hemorrhage",
    "myopic_fundus", "increased_cup_disc", "other"
]

ORDINAL_TASK = "DR_ICDR"

BINARY_METRICS = ["AUROC", "AUPRC", "Accuracy", "Balanced_Accuracy", "Precision", "Recall", "Specificity", "F1"]
ORDINAL_METRICS = ["AUROC", "Accuracy", "Balanced_Accuracy", "Macro_F1", "Weighted_F1", "QWK", "MAE"]

@dataclass
class Config:
    batch_size: int = 8
    num_workers: int = 8
    n_bootstrap: int = 100
    ci_level: float = 0.95
    resolutions: List[int] = field(default_factory=lambda: [392])
    rfg_path: str = "../res/PRD_checkpoint.pth"
    res_dir: str = "../res/"

    @property
    def comps(self) -> List[Tuple[int, int]]:
        return list(combinations(self.resolutions, 2))



TIMINGS = []
@contextmanager
def timer(name):
    start = perf_counter()
    try:
        yield
    finally:
        # This block is guaranteed to run even if the loop errors out or returns
        elapsed = perf_counter() - start
        print(f"{name}: {elapsed:.2f}s")
        TIMINGS.append((name, elapsed))
        