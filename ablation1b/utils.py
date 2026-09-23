

import numpy as np
from sklearn.metrics import precision_recall_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


    
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score
from scipy.stats import kruskal, chi2_contingency, fisher_exact, ttest_ind

from contextlib import contextmanager

@contextmanager
def track_compute(name, log_to_file=True):

    process = psutil.Process(os.getpid())

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------
    mem_before = process.memory_info().rss

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gpu_before = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_reserved_before = torch.cuda.memory_reserved() / (1024 ** 2)
    else:
        gpu_before = 0
        gpu_reserved_before = 0

    start_time = time.perf_counter()

    try:
        yield

    finally:

        # ----------------------------------------------------
        # AFTER
        # ----------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        mem_after = process.memory_info().rss

        if torch.cuda.is_available():
            gpu_after = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_reserved_after = torch.cuda.memory_reserved() / (1024 ** 2)
        else:
            gpu_after = 0
            gpu_reserved_after = 0

        elapsed = time.perf_counter() - start_time

        # ----------------------------------------------------
        # CPU
        # ----------------------------------------------------
        delta_cpu_mb = (
            mem_after - mem_before
        ) / (1024 ** 2)

        total_cpu_mb = (
            mem_after
        ) / (1024 ** 2)

        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------
        delta_gpu_mb = gpu_after - gpu_before
        delta_reserved_mb = (
            gpu_reserved_after - gpu_reserved_before
        )

        msg = (
            f"[{name}] "
            f"Time: {elapsed:.2f}s | "
            f"CPU: {total_cpu_mb:.0f} MiB "
            f"(Δ {delta_cpu_mb:+.0f}) | "
            f"GPU allocated: {gpu_after:.0f} MiB "
            f"(Δ {delta_gpu_mb:+.0f}) | "
            f"GPU reserved: {gpu_reserved_after:.0f} MiB "
            f"(Δ {delta_reserved_mb:+.0f})"
        )

        print(msg)

        if log_to_file:
            logging.info(msg)
            flush_log()

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def find_best_mcc_threshold_vectorized(probs, targets):
    # Convert inputs to numpy arrays just in case
    probs_arr = np.asarray(probs)
    targets_arr = np.asarray(targets)
    
    # 1. Generate grid thresholds (100 steps)
    thresholds = np.linspace(0.01, 0.99, 100)
    
    # 2. Broadcast into a 2D matrix: shape (len(probs), 100)
    preds = (probs_arr[:, None] >= thresholds).astype(int)
    targets_col = targets_arr[:, None]

    # 3. Vectorized calculation of all Confusion Matrix quadrants across all 100 thresholds
    tp = np.sum((preds == 1) & (targets_col == 1), axis=0).astype(np.float64)
    fp = np.sum((preds == 1) & (targets_col == 0), axis=0).astype(np.float64)
    fn = np.sum((preds == 0) & (targets_col == 1), axis=0).astype(np.float64)
    tn = np.sum((preds == 0) & (targets_col == 0), axis=0).astype(np.float64)

    # 4. Vectorized MCC Formula
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    # Avoid division by zero for edge thresholds (e.g., threshold=0.99 where TP+FP might be 0)
    denominator[denominator == 0] = 1e-6 
    
    mcc_scores = numerator / denominator
    
    # 5. Extract the best threshold
    best_idx = np.argmax(mcc_scores)
    return thresholds[best_idx]

def find_best_fbeta_threshold_vectorized(probs, targets, beta=0.5):
    probs_arr = np.asarray(probs)
    targets_arr = np.asarray(targets)
    
    # 1. Generate grid thresholds (100 steps)
    thresholds = np.linspace(0.01, 0.99, 100)
    
    # 2. Broadcast into a 2D matrix: shape (len(probs), 100)
    preds = (probs_arr[:, None] >= thresholds).astype(int)
    targets_col = targets_arr[:, None]

    # 3. Vectorized confusion matrix calculation (only positive class metrics are needed)
    tp = np.sum((preds == 1) & (targets_col == 1), axis=0).astype(np.float64)
    fp = np.sum((preds == 1) & (targets_col == 0), axis=0).astype(np.float64)
    fn = np.sum((preds == 0) & (targets_col == 1), axis=0).astype(np.float64)

    # 4. Vectorized Precision & Recall
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    # 5. Vectorized F-beta Formula
    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    
    # Handle absolute edge cases (e.g., threshold=0.99 with 0 predictions)
    denominator[denominator == 0] = 1e-6
    
    fbeta_scores = numerator / denominator
    
    # 6. Extract the best threshold
    best_idx = np.argmax(fbeta_scores)
    return thresholds[best_idx]

def find_thresholds_vectorized(probs, targets):
    thresholds = np.linspace(0.01, 0.99, 100)
    
    # Broadcast predictions into a 2D matrix: shape (len(probs), 100)
    preds = (probs_arr[:, None] >= thresholds).astype(int)
    targets_col = targets_arr[:, None]

    # Calculate True Positives, False Positives, False Negatives, True Negatives across all columns
    tp = np.sum((preds == 1) & (targets_col == 1), axis=0)
    fp = np.sum((preds == 1) & (targets_col == 0), axis=0)
    fn = np.sum((preds == 0) & (targets_col == 1), axis=0)
    tn = np.sum((preds == 0) & (targets_col == 0), axis=0)

    # Vectorized Youden's J
    j_scores = tp / (tp + fn + 1e-6) + tn / (tn + fp + 1e-6) - 1
    youden_thr = thresholds[np.argmax(j_scores)]

    # Vectorized Macro F1
    p1, r1 = tp / (tp + fp + 1e-6), tp / (tp + fn + 1e-6)
    f1_class1 = 2 * p1 * r1 / (p1 + r1 + 1e-6)
    
    p0, r0 = tn / (tn + fn + 1e-6), tn / (tn + fp + 1e-6)
    f1_class0 = 2 * p0 * r0 / (p0 * r0 + 1e-6)
    
    macro_f1 = (f1_class1 + f1_class0) / 2
    f1_thr = thresholds[np.argmax(macro_f1)]

    return f1_thr, youden_thr

def find_fbeta_threshold(probs, targets, beta=0.5):
    # Perfect for imbalanced data because it ignores True Negatives
    precision, recall, thresholds = precision_recall_curve(targets, probs)
    
    beta_sq = beta ** 2
    fbeta = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall + 1e-6)
    
    # Remove the last element of precision/recall because thresholds array is length N-1
    best_idx = np.argmax(fbeta[:-1]) 
    return thresholds[best_idx]

def specificity_score(y, q):
    tn, fp, _, _ = confusion_matrix(y, q, labels=[0, 1]).ravel()
    return np.nan if (tn + fp) == 0 else tn / (tn + fp)


def cka_similarity(X, Y):
    """Centered Kernel Alignment between two feature matrices."""
    # Center the features
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    K = X @ X.T
    L = Y @ Y.T
    hsic = np.trace(K @ L) / ((K.shape[0] - 1) ** 2)
    
    # Normalize to get CKA
    norm_K = np.sqrt(np.trace(K @ K))
    norm_L = np.sqrt(np.trace(L @ L))
    if norm_K == 0 or norm_L == 0:
        return 0.0
    return hsic / (norm_K * norm_L)
     