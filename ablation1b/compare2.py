import os, random, copy, argparse, yaml
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
import matplotlib.pyplot as plt
import torch, timm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, recall_score, precision_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve
from scipy.stats import kruskal, chi2_contingency, fisher_exact, ttest_ind
import time
import psutil
import logging, argparse
from contextlib import contextmanager

USERID = os.environ.get("USER", "").lower()
USERID = f"def-{USERID}-ab"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ============================================================
# SETUP
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(device)

ncores = len(os.sched_getaffinity(0)) // 2
pin_mem = (device == "cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--TK", type=str, default='D')
parser.add_argument("--ALPHA", type=float, default=.1)
parser.add_argument("--RES1", type=int, default=980)
parser.add_argument("--RES2", type=int, default=980)
parser.add_argument("--fold", type=int, default=1)
parser.add_argument("--two_stage", type=int, default=0)
parser.add_argument("--TEMPERATURE", type=float, default=1.0)

args, _ = parser.parse_known_args()

print('Arguments to compare.py:\n', args)

# ============================================================
# CONFIG
# ============================================================
@dataclass
class Config:
    BS: int = 8
    PATIENCE: int = 10
    ALPHA: float = 0.11
    CENTROID_CROP: bool = True
    LR: float = 1e-5
    MAX_EPOCHS: int = 1
    RES1: int = 980
    RES2: int = 980
    THRESHOLD_RULE = "mcc"
    res_dir: str = "./res/"
    vis_dir: str = "./res/"

    TEMPERATURE: float = 1.0

    two_stage: int = 0
    fold: int = 1
    TK: str = 'A'
    task: str = TK

    def checkpoint_filepath(self, method):
        pref = self.pref()
        return f"{pref}_{method}.pt"

    def pref(self):
        return f"{self.res_dir}{self.task}_fd{self.fold}_res{self.RES1}_bs{self.BS}_staged{self.two_stage}_T{self.TEMPERATURE}_alp{self.ALPHA}_crop{int(self.CENTROID_CROP)}_lr{self.LR}"


conf = Config(RES1=args.RES1, RES2=args.RES2, two_stage=args.two_stage, fold=args.fold, ALPHA=args.ALPHA, TK=args.TK)
pref = conf.pref()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MiB


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
        delta_cpu_mb = (mem_after - mem_before) / (1024 ** 2)
        total_cpu_mb = mem_after / (1024 ** 2)

        # ----------------------------------------------------
        # GPU
        # ----------------------------------------------------
        delta_gpu_mb = gpu_after - gpu_before
        delta_reserved_mb = gpu_reserved_after - gpu_reserved_before

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
    probs_arr = np.asarray(probs)
    targets_arr = np.asarray(targets)

    thresholds = np.linspace(0.01, 0.99, 100)

    preds = (probs_arr[:, None] >= thresholds).astype(int)
    targets_col = targets_arr[:, None]

    tp = np.sum((preds == 1) & (targets_col == 1), axis=0).astype(np.float64)
    fp = np.sum((preds == 1) & (targets_col == 0), axis=0).astype(np.float64)
    fn = np.sum((preds == 0) & (targets_col == 1), axis=0).astype(np.float64)
    tn = np.sum((preds == 0) & (targets_col == 0), axis=0).astype(np.float64)

    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    denominator[denominator == 0] = 1e-6

    mcc_scores = numerator / denominator

    best_idx = np.argmax(mcc_scores)
    return thresholds[best_idx]


def find_best_fbeta_threshold_vectorized(probs, targets, beta=0.5):
    probs_arr = np.asarray(probs)
    targets_arr = np.asarray(targets)

    thresholds = np.linspace(0.01, 0.99, 100)

    preds = (probs_arr[:, None] >= thresholds).astype(int)
    targets_col = targets_arr[:, None]

    tp = np.sum((preds == 1) & (targets_col == 1), axis=0).astype(np.float64)
    fp = np.sum((preds == 1) & (targets_col == 0), axis=0).astype(np.float64)
    fn = np.sum((preds == 0) & (targets_col == 1), axis=0).astype(np.float64)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    beta_sq = beta ** 2
    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    denominator[denominator == 0] = 1e-6

    fbeta_scores = numerator / denominator

    best_idx = np.argmax(fbeta_scores)
    return thresholds[best_idx]


def find_thresholds_vectorized(probs, targets):
    # FIX: ensure 1-D inputs so broadcasting below is (N, 100), not (N, N, 100)
    probs = np.asarray(probs).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    thresholds = np.linspace(0.01, 0.99, 100)

    preds = (probs[:, None] >= thresholds).astype(int)
    targets_col = targets[:, None]

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
    
    # FIX: denominator should be (p0 + r0), not (p0 * r0), to match the F1 formula  <-- cld
    f1_class0 = 2 * p0 * r0 / (p0 + r0 + 1e-6)

    macro_f1 = (f1_class1 + f1_class0) / 2
    f1_thr = thresholds[np.argmax(macro_f1)]

    return f1_thr, youden_thr


def find_fbeta_threshold(probs, targets, beta=0.5):
    # Perfect for imbalanced data because it ignores True Negatives
    precision, recall, thresholds = precision_recall_curve(targets, probs)

    beta_sq = beta ** 2
    fbeta = (1 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall + 1e-6)

    best_idx = np.argmax(fbeta[:-1])
    return thresholds[best_idx]


def specificity_score(y, q):
    tn, fp, _, _ = confusion_matrix(y, q, labels=[0, 1]).ravel()
    return np.nan if (tn + fp) == 0 else tn / (tn + fp)


def cka_similarity(X, Y):
    """Centered Kernel Alignment between two feature matrices."""
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    K = X @ X.T
    L = Y @ Y.T
    hsic = np.trace(K @ L) / ((K.shape[0] - 1) ** 2)

    norm_K = np.sqrt(np.trace(K @ K))
    norm_L = np.sqrt(np.trace(L @ L))
    if norm_K == 0 or norm_L == 0:
        return 0.0
    return hsic / (norm_K * norm_L)


def flush_log():
    for handler in logging.root.handlers:
        handler.flush()


logging.info(conf)


# Setup logging to save prints to a file
logging.basicConfig(
    filename=f'{pref}_experiment_benchmark.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'  # Overwrites old log each run
)

if 0:  # args.config is not None:
    config = load_config(args.config)
    DATA_DIR = args.data_dir if args.data_dir is not None else config["dataset"]["data_dir"]
    metadata_file = os.path.join(DATA_DIR, config["dataset"]["metadata_file"])
    IM_DIR = os.path.join(DATA_DIR, config["dataset"]["image_subdirectory"])
    CHKPATH = config.get("checkpoint_path", f"/project/{USERID}/HowRU/retinal/assets/retfoundgreen_statedict.pth")
    MAX_EPOCHS = config.get("max_epochs", MAX_EPOCHS)
    LR = config.get("lr", LR)
    TASK = config.get("TK", "D")
    conf.CENTROID_CROP = config.get("conf.CENTROID_CROP", conf.CENTROID_CROP)
else:

    if os.path.exists("/kaggle"):
        DATA_DIR = "/kaggle/input/datasets/andrewmvd/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/"
        IM_DIR = DATA_DIR + "/Training Images"
        CHKPATH = "rfg_statedict.pth"
    else:
        CHKPATH = f"/project/{USERID}/retfoundgreen_statedict.pth"
        DATA_DIR = f"/project/{USERID}/ODIR-5K/ODIR-5K/"
        IM_DIR = DATA_DIR + "/trn"
    metadata_file = DATA_DIR + "data.xlsx"

# ============================================================
# DATA
# ============================================================

odir_df = pd.read_excel(metadata_file)


def reformat_and_prepare_odir(df):
    target_cols = ["N", "D", "G", "C", "A", "H", "M", "O"]
    out = df[["ID", "Left-Fundus", "Right-Fundus"] + target_cols].copy()
    out = out.rename(columns={"ID": "patient_id"})
    for col in target_cols:
        out[col] = out[col].astype(int)
    return out, target_cols


def split_dataset_by_patient(df, args):
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    gkf = GroupKFold(n_splits=6)

    for fold, (train_val_idx, test_idx) in enumerate(gkf.split(df, groups=df["patient_id"])):
        if fold == args.fold:  # second fold, since indexing starts at 0
            break

    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gkf = GroupKFold(n_splits=6)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_val_df, groups=train_val_df["patient_id"])):
        if fold == args.fold:  # second fold, since indexing starts at 0
            break

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    tr_patients = set(train_df["patient_id"])
    va_patients = set(val_df["patient_id"])
    te_patients = set(test_df["patient_id"])

    assert tr_patients.isdisjoint(va_patients), "Patient overlap: train/val"
    assert tr_patients.isdisjoint(te_patients), "Patient overlap: train/test"
    assert va_patients.isdisjoint(te_patients), "Patient overlap: val/test"

    print("No patient overlap \u2713")

    return train_df, val_df, test_df


def filter_existing_images(df, image_dir):
    left = df["Left-Fundus"].apply(lambda x: os.path.exists(os.path.join(image_dir, str(x))))
    right = df["Right-Fundus"].apply(lambda x: os.path.exists(os.path.join(image_dir, str(x))))
    return df[left & right].reset_index(drop=True)


# ============================================================
# DATASET
# ============================================================

class CFPDataset(Dataset):

    def __init__(self, dataframe, target_cols, image_dir, CENTROID_CROP=True, target_size=(392, 392), image_col_left="Left-Fundus", image_col_right="Right-Fundus", single_image=False, file_format=None):
        # FIX: honor the CENTROID_CROP argument passed to this instance instead of
        # silently overriding it with the global conf value.
        self.CENTROID_CROP = CENTROID_CROP
        self.target_cols = [target_cols] if isinstance(target_cols, str) else list(target_cols)
        self.image_dir = image_dir
        self.target_size = target_size
        self.image_col_left = image_col_left
        self.image_col_right = image_col_right
        self.single_image = single_image
        self.file_format = file_format
        self.samples = dataframe.copy().reset_index(drop=True)
        for col in self.target_cols:
            self.samples[col] = pd.to_numeric(self.samples[col], errors="coerce").fillna(0).astype(np.float32)

    def _apply_CENTROID_CROP(self, img):
        _, h, w = img.shape
        gray = img.mean(dim=0)
        idx = torch.nonzero(gray > 0.05)
        if idx.numel() == 0:
            return img
        mins, _ = idx.min(dim=0)
        maxs, _ = idx.max(dim=0)
        ymin, xmin = mins.tolist()
        ymax, xmax = maxs.tolist()
        diameter = min(ymax - ymin + 1, xmax - xmin + 1)
        half = diameter // 2
        cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
        ys, ye = cy - half, cy + half
        xs, xe = cx - half, cx + half
        pt, pb = max(0, -ys), max(0, ye - h)
        pl, pr = max(0, -xs), max(0, xe - w)
        ys, ye = max(0, ys), min(h, ye)
        xs, xe = max(0, xs), min(w, xe)
        crop = img[:, ys:ye, xs:xe]
        if pt or pb or pl or pr:
            crop = TF.pad(crop, [pl, pt, pr, pb], fill=0)
        return crop

    def _load_image(self, filename):
        if self.file_format is not None:
            filename = f"{filename}.{self.file_format}"
        path = os.path.join(self.image_dir, str(filename))
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
            img = TF.to_tensor(img)
            if self.CENTROID_CROP:
                img = self._apply_CENTROID_CROP(img)
        except Exception:
            img = torch.zeros(3, self.target_size[0], self.target_size[1])

        img = TF.resize(img, self.target_size)
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        label = torch.tensor(row[self.target_cols].values.astype(np.float32))
        if self.single_image == 'left':
            x = self._load_image(row[self.image_col_left])
            return x, label, int(idx)
        elif self.single_image == 'right':
            x = self._load_image(row[self.image_col_right])
            return x, label, int(idx)
        else:
            left = self._load_image(row[self.image_col_left])
            right = self._load_image(row[self.image_col_right])
            return left, right, label, int(idx)


def show_batch(D, n=8, out_file=None):
    try:
        sample_batch_imgs, sample_batch_imgs2, sample_labels, sample_indices = D
    except Exception:
        sample_batch_imgs2 = None
        sample_batch_imgs, sample_labels, sample_indices = D

    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    axes = axes.flatten()
    cnt = 0

    show_i = np.where(sample_labels)[0]
    try:
        S = [show_i[0]] + list(np.arange(n - 1))
    except Exception:
        S = list(np.arange(n - 1))

    for i in S:
        img = sample_batch_imgs[i].permute(1, 2, 0).cpu().numpy()
        lab = sample_labels[i].numpy()
        pid = sample_indices[i]

        axes[cnt].imshow(np.clip(img, 0, 1))
        axes[cnt].set_title(f"Patient {pid} | outcome={lab}")
        axes[cnt].axis("off")
        cnt += 1
        try:
            img2 = sample_batch_imgs2[i]
            if img2 is not None:
                img2 = img2.permute(1, 2, 0).cpu().numpy()
                axes[cnt].imshow(np.clip(img2, 0, 1))
                axes[cnt].set_title(f"Patient {pid} | outcome={lab} (view2)")
                axes[cnt].axis("off")
                cnt += 1
        except Exception as e:
            print(f"Skipping view 2 for sample {i}: {e}")
        if cnt >= len(axes):
            break

    for j in range(cnt, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if out_file:
        plt.savefig(f"{out_file}.png", bbox_inches="tight")
    plt.show()


# ============================================================
# BACKBONE
# ============================================================

def get_backbone(RES1, RES2):
    model = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        img_size=(RES1, RES2),
        num_classes=0,
        dynamic_img_size=True
    )

    checkpoint = torch.load(CHKPATH, map_location="cpu")
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    if "pos_embed" in state_dict:
        pos_embed = state_dict["pos_embed"]
        B, N, C = pos_embed.shape
        old_grid = int(N ** 0.5)
        new_grid = model.patch_embed.grid_size[0]

        pos_embed = pos_embed.reshape(B, old_grid, old_grid, C).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(B, new_grid * new_grid, C)

        state_dict["pos_embed"] = pos_embed

    model.load_state_dict(state_dict, strict=False)
    model.global_pool = "avg"

    return model


# ============================================================
# SINGLE-EYE NETWORK
# ============================================================

class SingleEyeNet(nn.Module):

    def __init__(self, backbone, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.classifier(z)
        return logits, z


# ============================================================
# PRIVILEGED TEACHER
#
# Teacher receives both eyes and learns directly from labels.
#
# z_teacher = bilateral representation obtained from both eyes,
# formed here as the mean of the two eye embeddings.
# ============================================================

class PrivilegedTeacher(nn.Module):

    def __init__(self, backbone, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, left, right):
        z_left = self.backbone(left)
        z_right = self.backbone(right)

        z_teacher = (z_left + z_right) / 2.0
        # z_teacher = torch.maximum(z_left, z_right)
        logits = self.classifier(z_teacher)

        return logits, z_teacher


# ============================================================
# PROPOSED PRIVILEGED DISTILLATION MODEL
#
# Teacher and student have independent parameters.
# Teacher learns from paired eyes.
# Student learns from individual eyes.
# ============================================================

class PrivilegedDistillation(nn.Module):

    def __init__(self, student_backbone, teacher_backbone, num_classes=1):
        super().__init__()

        self.student = SingleEyeNet(student_backbone, num_classes)
        self.teacher = PrivilegedTeacher(teacher_backbone, num_classes)

    def student_forward(self, x):
        return self.student(x)

    def teacher_forward(self, left, right):
        return self.teacher(left, right)


# ============================================================
# LOSS
# ============================================================

def make_pos_weight(df, task):
    n_positive = float(df[task].values.sum()) + 1e-10
    n_negative = float(len(df) - n_positive)
    return torch.tensor([n_negative / max(n_positive, 1.0)], device=DEVICE)


# ============================================================
# BASELINE TRAINING
#
# IMPORTANT:
# The baseline explicitly trains on BOTH eyes.
#
# A batch of N patients becomes 2N single-eye examples:
#     [L1 ... LN, R1 ... RN]
#
# Therefore the baseline sees exactly the same L/R images
# available to the proposed student.
# ============================================================

def train_baseline(model, train_loader, val_loader, trn_df, conf):

    model.to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=conf.LR)
    pos_weight = make_pos_weight(trn_df, conf.task)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = np.inf
    patience_counter = 0
    checkpoint = conf.checkpoint_filepath("baseline")

    for epoch in range(1, conf.MAX_EPOCHS + 1):

        model.train()
        train_loss = 0.0

        for left, right, targets, _ in train_loader:

            left = left.to(DEVICE, non_blocking=True)
            right = right.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            # Same paired dataset, but baseline treats eyes independently.
            x = torch.cat([left, right], dim=0)
            y = torch.cat([targets, targets], dim=0)

            logits, _ = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss, val_auc = evaluate_validation_single_eye(model, val_loader, criterion)

        print(f"[BASELINE] Epoch {epoch:02d} | Train={train_loss:.4f} | Val={val_loss:.4f} | AUROC={val_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint)
            print(epoch, 'saving checkpoint at', checkpoint)
        else:
            patience_counter += 1
            if patience_counter >= conf.PATIENCE:
                print("[BASELINE] Early stopping.")
                break

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    return model


# ============================================================
# PROPOSED TRAINING
#
# Teacher:
#   L + R -> privileged bilateral representation
#
# Student:
#   L -> prediction + distillation
#   R -> prediction + distillation
#
# Both student eyes are used every batch, ensuring that the
# student sees exactly the same eye-image population as baseline.
# ============================================================

def train_privileged(model, train_loader, val_loader, trn_df, conf):

    model.to(DEVICE)

    optimizer = optim.AdamW(
        list(model.student.parameters()) + list(model.teacher.parameters()),
        lr=conf.LR
    )

    pos_weight = make_pos_weight(trn_df, conf.task)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = np.inf
    patience_counter = 0
    checkpoint = conf.checkpoint_filepath("privileged_distillation")

    for epoch in range(1, conf.MAX_EPOCHS + 1):

        model.student.train()
        model.teacher.train()

        train_loss = 0.0

        for left, right, targets, _ in train_loader:

            left = left.to(DEVICE, non_blocking=True)
            right = right.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            # ------------------------------------------------
            # TEACHER: privileged paired-eye learning
            # ------------------------------------------------

            teacher_logits, z_teacher = model.teacher_forward(left, right)

            loss_teacher = criterion(teacher_logits, targets)

            # ------------------------------------------------
            # STUDENT: both eyes independently
            #
            # This is deliberately NOT random selection.
            # Both L and R are used so the baseline and proposed
            # student receive the same number of eye images.
            # ------------------------------------------------

            x_student = torch.cat([left, right], dim=0)
            y_student = torch.cat([targets, targets], dim=0)

            student_logits, z_student = model.student_forward(x_student)

            # Repeat privileged representation for L and R.
            z_teacher_student = torch.cat([z_teacher, z_teacher], dim=0)

            loss_student_cls = criterion(student_logits, y_student)

            loss_distill = F.mse_loss(z_student, z_teacher_student.detach())

            # Student learns classification + privileged representation.
            loss_student = loss_student_cls + conf.ALPHA * loss_distill

            # Teacher also learns directly from its privileged input.
            loss = loss_teacher + loss_student

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_loss, val_auc = evaluate_validation_single_eye(model.student, val_loader, criterion)

        print(f"[PRIVILEGED] Epoch {epoch:02d} | Train={train_loss:.4f} | Val={val_loss:.4f} | AUROC={val_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            #torch.save(model.student.state_dict(), checkpoint)

            torch.save({                
                "student": model.student.state_dict(),               
                "teacher": model.teacher.state_dict(),          
            }, checkpoint)                                    
            print(epoch, 'saving checkpoint at', checkpoint)
        else:
            patience_counter += 1
            if patience_counter >= conf.PATIENCE:
                print("[PRIVILEGED] Early stopping.")
                break

    #model.student.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    checkpoint = torch.load(checkpoint, map_location=DEVICE)
    model.student.load_state_dict(checkpoint["student"])
    model.teacher.load_state_dict(checkpoint["teacher"])    
    return model


def train_privileged_logit_kd(model, train_loader, val_loader, trn_df, conf):
    """Trains a single-eye student via Temperature Logit Distillation from a
    privileged bilateral teacher.

    NOTE: this function only FREEZES the teacher; it does not train it first.
    For genuine two-stage distillation, train/load a converged
    `PrivilegedTeacher` checkpoint (e.g. via `train_baseline`-style paired-eye
    supervision) and load its weights into `model.teacher` BEFORE calling this
    function. Calling it on a freshly-initialized teacher head means the
    student distills from an untrained teacher, which will not behave like
    real logit distillation.

    Key Benefits over Feature MSE (once the teacher is actually pretrained):
    1. Prevents feature hallucination (backbone remains a clean single-eye feature extractor).
    2. Transfers privileged bilateral risk/uncertainty directly to student confidence limits.
    3. Works natively with linear probes on frozen backbones.
    """
    model.to(DEVICE)

    # Hyperparameters
    T = getattr(conf, "TEMPERATURE", 2.0)  # Temperature for softening logits
    alpha = getattr(conf, "ALPHA", 0.5)  # Balance between hard BCE and soft KD loss

    # 1. Freeze Teacher (Two-Stage Distillation)
    model.teacher.eval()
    for param in model.teacher.parameters():
        param.requires_grad = False

    # 2. Train Student Only
    model.student.train()
    optimizer = optim.AdamW(model.student.parameters(), lr=conf.LR)

    pos_weight = make_pos_weight(trn_df, conf.task)
    criterion_hard = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = np.inf
    patience_counter = 0
    checkpoint = conf.checkpoint_filepath("privileged_logit_kd")

    for epoch in range(1, conf.MAX_EPOCHS + 1):
        model.student.train()

        running_total_loss = 0.0
        running_cls_loss = 0.0
        running_kd_loss = 0.0

        for left, right, targets, _ in train_loader:
            left = left.to(DEVICE, non_blocking=True)
            right = right.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            # -----------------------------------------------------------------
            # 1. TEACHER FORWARD (Privileged paired-eye targets)
            # -----------------------------------------------------------------
            with torch.no_grad():
                teacher_logits, _ = model.teacher_forward(left, right)
                teacher_logits_rep = torch.cat([teacher_logits, teacher_logits], dim=0)

            # -----------------------------------------------------------------
            # 2. STUDENT FORWARD (Single-eye inputs)
            # -----------------------------------------------------------------
            x_student = torch.cat([left, right], dim=0)
            y_student = torch.cat([targets, targets], dim=0)

            student_logits, _ = model.student_forward(x_student)

            # -----------------------------------------------------------------
            # 3. HARD LOSS: Standard Ground-Truth BCE
            # -----------------------------------------------------------------
            loss_hard = criterion_hard(student_logits, y_student)

            # -----------------------------------------------------------------
            # 4. SOFT LOSS: Temperature-Scaled Logit Distillation
            # -----------------------------------------------------------------
            soft_teacher_probs = torch.sigmoid(teacher_logits_rep / T)
            soft_student_logits = student_logits / T

            loss_soft = (T ** 2) * F.binary_cross_entropy_with_logits(soft_student_logits, soft_teacher_probs)

            loss = ((1.0 - alpha) * loss_hard) + (alpha * loss_soft)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_total_loss += loss.item()
            running_cls_loss += loss_hard.item()
            running_kd_loss += loss_soft.item()

        train_total_loss = running_total_loss / len(train_loader)
        train_cls_loss = running_cls_loss / len(train_loader)
        train_kd_loss = running_kd_loss / len(train_loader)

        val_loss, val_auc = evaluate_validation_single_eye(model.student, val_loader, criterion_hard)

        print(
            f"[LOGIT-KD] Epoch {epoch:02d} | "
            f"Train Total={train_total_loss:.4f} | "
            f"Hard Cls={train_cls_loss:.4f} | "
            f"Soft KD={train_kd_loss:.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Val AUROC={val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print('Saving checkpoint with min. val_loss')
            #torch.save(model.student.state_dict(), checkpoint)
            torch.save({
                "student": model.student.state_dict(),
                "teacher": model.teacher.state_dict(),
            }, checkpoint)
                        
        else:
            patience_counter += 1
            if patience_counter >= conf.PATIENCE:
                print("[LOGIT-KD] Early stopping triggered.")
                break

    #model.student.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    checkpoint = torch.load(checkpoint, map_location=DEVICE)
    model.student.load_state_dict(checkpoint["student"])
    model.teacher.load_state_dict(checkpoint["teacher"])    
    
    print('Loaded checkpoint with min. val_loss')
    return model


# ============================================================
# VALIDATION
# ============================================================

def evaluate_validation_single_eye(model, loader, criterion):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0
    with torch.no_grad():
        for left, right, targets, _ in loader:
            left = left.to(DEVICE, non_blocking=True)
            right = right.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            # Evaluate BOTH eyes.
            x = torch.cat([left, right], dim=0)
            y = torch.cat([targets, targets], dim=0)

            logits, _ = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()

            ys.append(y[:, 0].cpu().numpy())
            ps.append(torch.sigmoid(logits[:, 0]).cpu().numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    return total_loss / len(loader), auc


# ============================================================
# PREDICTION
# ============================================================
def get_single_eye_predictions(model, loader, eye):
    model.eval()
    ys, ps, idxs = [], [], []
    with torch.no_grad():
        for batch in loader:
            # Single-eye dataset (e.g., BRSet) -> batch has 3 items
            if len(batch) == 3:
                x, targets, idx = batch
            else:
                # Paired dataset (ODIR) -> batch has 4 items
                left, right, targets, idx = batch
                if eye == "left":
                    x = left
                elif eye == "right":
                    x = right
                else:
                    # If eye == "single", default to left (or handle externally)
                    x = left

            x = x.to(DEVICE, non_blocking=True)
            logits, _ = model(x)
            ys.append(targets[:, 0].cpu().numpy())
            ps.append(torch.sigmoid(logits[:, 0]).cpu().numpy())
            idxs.append(idx.cpu().numpy())

    return np.concatenate(ys), np.concatenate(ps), np.concatenate(idxs)


# ============================================================
# THRESHOLD
# ============================================================
def select_optimal_threshold(conf, y_true, y_prob):
    if conf.THRESHOLD_RULE == "mcc":
        return find_best_mcc_threshold_vectorized(y_prob, y_true)

    f1_thr, youden_thr = find_thresholds_vectorized(y_prob, y_true)

    return f1_thr if conf.THRESHOLD_RULE == "f1" else youden_thr


# ============================================================
# TEST
# ============================================================
def evaluate_test(model, loader, eye, threshold, n_bootstraps=1000, alpha=0.05):
    """
    Evaluates the model and returns point estimates + 95% Confidence Intervals
    for all metrics using the Bootstrap method.
    """
    y_true, y_prob, _ = get_single_eye_predictions(model, loader, eye)
    pred = y_prob >= threshold

    # -------------------- Point Estimates --------------------
    metrics = {
        "Eye": eye,
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Acc": balanced_accuracy_score(y_true, pred),
        "F1": f1_score(y_true, pred, zero_division=0),
        "F1 Macro": f1_score(y_true, pred, average='macro', zero_division=0),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # -------------------- Bootstrapped CIs --------------------
    rng = np.random.RandomState(args.seed)
    boot_records = {k: [] for k in metrics.keys() if k != "Eye"}
    n = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        y_t = y_true[idx]
        y_p = y_prob[idx]
        y_pred_b = y_p >= threshold

        if len(np.unique(y_t)) < 2:
            continue

        try:
            boot_records["AUROC"].append(roc_auc_score(y_t, y_p))
        except ValueError:
            pass
        try:
            boot_records["AUPRC"].append(average_precision_score(y_t, y_p))
        except ValueError:
            pass

        boot_records["Accuracy"].append(accuracy_score(y_t, y_pred_b))
        boot_records["Balanced Acc"].append(balanced_accuracy_score(y_t, y_pred_b))
        boot_records["F1"].append(f1_score(y_t, y_pred_b, zero_division=0))
        boot_records["F1 Macro"].append(f1_score(y_t, y_pred_b, average='macro', zero_division=0))
        boot_records["Precision"].append(precision_score(y_t, y_pred_b, zero_division=0))
        boot_records["Recall"].append(recall_score(y_t, y_pred_b, zero_division=0))

        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_t, y_pred_b).ravel()
        boot_records["FPR"].append(fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0)

    cis = {}
    for k, v in boot_records.items():
        if v:
            lower = np.percentile(v, 100 * (alpha / 2))
            upper = np.percentile(v, 100 * (1 - alpha / 2))
            cis[k] = (lower, upper)
        else:
            cis[k] = (np.nan, np.nan)

    return metrics, cis


def run_inference(model, loader, eye):
    """
    Runs the model on the entire loader for the specified eye.
    Returns:
        y_true: ground truth labels (1-D)
        y_prob: predicted probabilities
        indices: sample indices matching the dataset
    """
    model = model.to(DEVICE)
    model.eval()
    features, ys, ps, idxs = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, targets, idx = batch
                x = x.to(DEVICE, non_blocking=True)
                try:
                    logits, feats = model(x)
                except Exception:
                    logits, feats = model(x, x)  # teacher sees identical inputs separately
            else:
                left, right, targets, idx = batch
                try:  # Try treating it as a single-eye model
                    x = left if eye == "left" else right
                    x = x.to(DEVICE, non_blocking=True)
                    logits, feats = model(x)
                except Exception:  # Fallback: Treat as a dual-eye model if single-eye fails
                    x1 = left.to(DEVICE, non_blocking=True)
                    x2 = right.to(DEVICE, non_blocking=True)
                    logits, feats = model(x1, x2)

            # FIX: flatten targets to 1-D. `targets` has shape (B, 1) since a
            # single task column is used; keeping it 2-D silently corrupted
            # downstream threshold selection (see find_thresholds_vectorized).
            ys.append(targets[:, 0].cpu().numpy())
            ps.append(torch.sigmoid(logits[:, 0]).cpu().numpy())
            idxs.append(idx.cpu().numpy())
            features.append(feats.detach().cpu().numpy())

    return np.concatenate(ys), np.concatenate(ps), np.concatenate(idxs)


def bootstrapped_metrics(y_true, y_prob, threshold, eye="unknown", n_bootstraps=1000, alpha=0.05):
    """
    Computes metrics and bootstrapped CIs from pre-computed predictions.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    pred = y_prob >= threshold

    metrics = {
        "Eye": eye,
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Acc": balanced_accuracy_score(y_true, pred),
        "F1": f1_score(y_true, pred, zero_division=0),
        "F1 Macro": f1_score(y_true, pred, average='macro', zero_division=0),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Specificity": specificity_score(y_true, pred),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    metrics["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    rng = np.random.RandomState(args.seed)
    boot_records = {k: [] for k in metrics.keys() if k not in ["Eye", "threshold"]}
    n = len(y_true)

    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        y_t = y_true[idx]
        y_p = y_prob[idx]
        y_pred_b = y_p >= threshold

        if len(np.unique(y_t)) < 2:
            continue

        try:
            boot_records["AUROC"].append(roc_auc_score(y_t, y_p))
        except ValueError:
            pass
        try:
            boot_records["AUPRC"].append(average_precision_score(y_t, y_p))
        except ValueError:
            pass

        boot_records["Accuracy"].append(accuracy_score(y_t, y_pred_b))
        boot_records["Balanced Acc"].append(balanced_accuracy_score(y_t, y_pred_b))
        boot_records["F1"].append(f1_score(y_t, y_pred_b, zero_division=0))
        boot_records["F1 Macro"].append(f1_score(y_t, y_pred_b, average='macro', zero_division=0))
        boot_records["Precision"].append(precision_score(y_t, y_pred_b, zero_division=0))
        boot_records["Recall"].append(recall_score(y_t, y_pred_b, zero_division=0))

        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_t, y_pred_b).ravel()
        boot_records["FPR"].append(fp_b / (fp_b + tn_b) if (fp_b + tn_b) > 0 else 0.0)

    cis = {}
    for k, v in boot_records.items():
        if v:
            lower = np.percentile(v, 100 * (alpha / 2))
            upper = np.percentile(v, 100 * (1 - alpha / 2))
            cis[k] = (lower, upper)
        else:
            cis[k] = (np.nan, np.nan)

    return metrics, cis


def get_predictions_df(y_true, y_prob, indices, threshold, loader, eye, dataset_name="ODIR"):
    """
    Builds a DataFrame from pre-computed predictions.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)

    dataset = loader.dataset
    df_samples = dataset.samples.iloc[indices].reset_index(drop=True)

    out_df = pd.DataFrame({
        "patient_id": df_samples.get("patient_id", df_samples.index),
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "threshold": threshold,
        "eye": eye,
        "dataset": dataset_name
    })

    if "Left-Fundus" in df_samples.columns and eye in ["left", "single"]:
        out_df["image_id"] = df_samples["Left-Fundus"]
    elif "Right-Fundus" in df_samples.columns and eye == "right":
        out_df["image_id"] = df_samples["Right-Fundus"]
    elif "image_id" in df_samples.columns:
        out_df["image_id"] = df_samples["image_id"]
    else:
        out_df["image_id"] = df_samples.index
    return out_df


def format_metric(name, value, ci):
    return f"{name:18s} {value:.4f}  [{ci[0]:.4f} \u2013 {ci[1]:.4f}]"


def log_model_params(name, model):
    count = sum(p.numel() for p in model.parameters())
    size_mb = (count * 4) / (1024 * 1024)  # fp32 = 4 bytes
    msg = f"[{name}] Parameters: {count:,} | Estimated Size: {size_mb:.2f} MiB"
    print(msg)
    logging.info(msg)
    flush_log()
    return count


def define_brset_splits(df):
    left_df = df[df.exam_eye == 2].reset_index(drop=True)
    right_df = df[df.exam_eye == 1].reset_index(drop=True)

    def split(cfp_df):
        cfp_df["age_group"] = pd.cut(cfp_df['patient_age'], bins=[-np.inf, 39, 49, 59, 69, 79, np.inf], labels=["<40", "40-49", "50-59", "60-69", "70-79", "80+"])
        quality_vars = ["focus", "illumination", "image_field", "artifacts"]

        rng = np.random.default_rng(args.seed)
        canon_patients = set(cfp_df.loc[cfp_df.camera == "Canon CR", "patient_id"])
        nikon = cfp_df[(cfp_df.camera == "NIKON NF5050") & ~cfp_df.patient_id.isin(canon_patients)].copy()

        secondary_patients, external_patients = set(), set()

        secondary_patients = set(nikon[nikon.image_field == 2].patient_id)  # 807
        secondary_df = nikon[nikon.patient_id.isin(secondary_patients)]
        external_df = nikon[~nikon.patient_id.isin(secondary_patients)]
        canon_df = cfp_df[cfp_df.patient_id.isin(canon_patients)]

        overlap = set(nikon["patient_id"]) & set(canon_df["patient_id"])
        print("Number of overlapping patient IDs:", len(overlap))
        overlap = set(secondary_df["patient_id"]) & set(external_df["patient_id"])
        print("Number of overlapping patient IDs:", len(overlap))

        return canon_df, secondary_df, external_df

    lcanon_df, lsec_df, lext_df = split(left_df)
    rcanon_df, rsec_df, rext_df = split(right_df)

    ext_loaders = {}

    side = 'left'
    ld0 = CFPDataset(lcanon_df, target_cols=conf.task, image_dir=CFP_IMAGE_DIR, image_col_left="image_id", image_col_right='image_id', single_image=side, file_format="jpg", CENTROID_CROP=conf.CENTROID_CROP)
    ld1 = CFPDataset(lsec_df, target_cols=conf.task, image_dir=CFP_IMAGE_DIR, image_col_left="image_id", image_col_right='image_id', single_image=side, file_format="jpg", CENTROID_CROP=conf.CENTROID_CROP)
    ld2 = CFPDataset(lext_df, target_cols=conf.task, image_dir=CFP_IMAGE_DIR, image_col_left="image_id", image_col_right='image_id', single_image=side, file_format="jpg", CENTROID_CROP=conf.CENTROID_CROP)
    ext_loaders[0, side] = DataLoader(ld0, batch_size=conf.BS, shuffle=True, num_workers=ncores, pin_memory=pin_mem)
    ext_loaders[1, side] = DataLoader(ld1, batch_size=conf.BS, shuffle=True, num_workers=ncores, pin_memory=pin_mem)
    ext_loaders[2, side] = DataLoader(ld2, batch_size=conf.BS, shuffle=True, num_workers=ncores, pin_memory=pin_mem)

    side = 'right'
    rd0 = CFPDataset(rcanon_df, target_cols=conf.task, image_dir=CFP_IMAGE_DIR, image_col_left="image_id", image_col_right='image_id', single_image=side, file_format="jpg", CENTROID_CROP=conf.CENTROID_CROP)
    rd1 = CFPDataset(rsec_df, target_cols=conf.task, image_dir=CFP_IMAGE_DIR, image_col_left="image_id", image_col_right='image_id', single_image=side, file_format="jpg", CENTROID_CROP=conf.CENTROID_CROP)
    rd2 = CFPDataset(rext_df, target_cols=conf.task, image_dir=CFP_IMAGE_DIR, image_col_left="image_id", image_col_right='image_id', single_image=side, file_format="jpg", CENTROID_CROP=conf.CENTROID_CROP)
    ext_loaders[0, side] = DataLoader(rd0, batch_size=conf.BS, shuffle=True, num_workers=ncores, pin_memory=pin_mem)
    ext_loaders[1, side] = DataLoader(rd1, batch_size=conf.BS, shuffle=True, num_workers=ncores, pin_memory=pin_mem)
    ext_loaders[2, side] = DataLoader(rd2, batch_size=conf.BS, shuffle=True, num_workers=ncores, pin_memory=pin_mem)

    return lcanon_df, lsec_df, lext_df, rcanon_df, rsec_df, rext_df, ext_loaders


def image_exists(filename):
    from pathlib import Path
    filename = f'{CFP_DATA_ROOT}/fundus_photos/{filename}.jpg'
    if not isinstance(filename, str):
        return False
    return Path(filename).is_file()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    set_all_seeds(args.seed)

    Models, preds, thresholds, cis, metrics = {}, {}, {}, {}, {}
    sides = ['left', 'right']
    MODES2 = ['baseline', 'student', 'teacher']

    # ============== Load ext validation set ==============
    try:
        CFP_DATA_ROOT = f'/project/{USERID}/physionet.org/files/brazilian-ophthalmological/1.0.2'
        CFP_IMAGE_DIR = os.path.join(CFP_DATA_ROOT, "fundus_photos")
        ext_df = pd.read_csv(os.path.join(CFP_DATA_ROOT, "label_brset.csv"))
    except Exception:  # running on Kaggle
        conf.res_dir = './'
        conf.vis_dir = './'
        CFP_DATA_ROOT = '/kaggle/input/datasets/tanzinabdul/fundus-patientwise-split/dataset_split/train/'
        CFP_IMAGE_DIR = '/kaggle/input/datasets/tanzinabdul/fundus-patientwise-split/dataset_split/train/images'
        ext_df = pd.read_csv('/kaggle/input/datasets/tanzinabdul/fundus-patientwise-split/dataset_split/train/train.csv')

    # harmonize labels
    ext_df['D'] = ext_df.diabetic_retinopathy
    ext_df['A'] = ext_df.amd
    ext_df['M'] = ext_df.myopic_fundus

    with track_compute("Reading external dataset + generating figure of samples"):
        DSET = 'brset'
        lcanon_df, lsec_df, lext_df, rcanon_df, rsec_df, rext_df, ext_loaders = define_brset_splits(ext_df)

        D = next(iter(ext_loaders[0, 'left']))
        show_batch(D, 8, f'{conf.task}_{DSET}')

    # ============== Load develop set ==============
    df, target_cols = reformat_and_prepare_odir(odir_df)
    trn_df, val_df, tst_df = split_dataset_by_patient(df, args)
    trn_df = filter_existing_images(trn_df, IM_DIR)
    val_df = filter_existing_images(val_df, IM_DIR)
    tst_df = filter_existing_images(tst_df, IM_DIR)
    print(f"Train patients: {len(trn_df)} | Val patients: {len(val_df)} | Test patients: {len(tst_df)}")
    print(f"Train eye images: {2 * len(trn_df)} | Val eye images: {2 * len(val_df)} | Test eye images: {2 * len(tst_df)}")

    with track_compute("Reading DEV dataset + generating figure of samples"):
        train_loader = DataLoader(CFPDataset(trn_df, conf.task, IM_DIR, CENTROID_CROP=conf.CENTROID_CROP), batch_size=conf.BS, shuffle=True, num_workers=ncores, pin_memory=pin_mem)
        val_loader = DataLoader(CFPDataset(val_df, conf.task, IM_DIR, CENTROID_CROP=conf.CENTROID_CROP), batch_size=conf.BS, shuffle=False, num_workers=ncores, pin_memory=pin_mem)
        test_loader = DataLoader(CFPDataset(tst_df, conf.task, IM_DIR, CENTROID_CROP=conf.CENTROID_CROP), batch_size=conf.BS, shuffle=False, num_workers=ncores, pin_memory=pin_mem)
        D = next(iter(test_loader))
        show_batch(D, 8, f'{conf.task}_ODIR_tst')

    # ========================================================
    # BASELINE
    # ========================================================
    print("\n" + "=" * 70)
    print("BASELINE: INDEPENDENT SINGLE-EYE MODEL")
    print("=" * 70)
    MODE = "baseline"
    mem_before = get_memory_usage()
    print(f"Memory before loading baseline: {mem_before:.2f} MiB")
    Models[MODE] = SingleEyeNet(get_backbone(conf.RES1, conf.RES2), num_classes=1)
    print(f"Memory after loading baseline: {get_memory_usage():.2f} MiB")
    print(f"Model footprint: {get_memory_usage() - mem_before:.2f} MiB")

    try:
        checkpoint_data = torch.load(conf.checkpoint_filepath(MODE), map_location=DEVICE)
        state_dict = checkpoint_data["state_dict"] if "state_dict" in checkpoint_data else checkpoint_data
        Models[MODE].load_state_dict(state_dict)
        print('loaded checkpoint', conf.checkpoint_filepath(MODE))
    except FileNotFoundError:
        with track_compute(f"Training {MODE}"):
            Models[MODE] = train_baseline(Models[MODE], train_loader, val_loader, trn_df, conf)

    # ============================================================
    # Privileged Distillation
    # ============================================================
    print("\n" + "=" * 70)
    print("PROPOSED: PRIVILEGED DISTILLATION")
    print("=" * 70)
    MODE = "student"
    try:
        student_backbone = get_backbone(conf.RES1, conf.RES2)
        teacher_backbone = get_backbone(conf.RES1, conf.RES2)
        privileged_model = PrivilegedDistillation(student_backbone, teacher_backbone, num_classes=1)        
        
        checkpoint = torch.load(conf.checkpoint_filepath(MODE), map_location=DEVICE)
                
        Models[MODE] = SingleEyeNet(get_backbone(conf.RES1, conf.RES2), num_classes=1)  # The saved checkpoint is the student (SingleEyeNet)
        Models[MODE].load_state_dict(checkpoint['student'])
        privileged_model.teacher.load_state_dict(checkpoint["teacher"])    
        
    except FileNotFoundError:
        with track_compute(f"Training {MODE}"):
            mem_before = get_memory_usage()
            print(f"Memory before loading privileged_model: {mem_before:.2f} MiB")
            student_backbone = get_backbone(conf.RES1, conf.RES2)
            teacher_backbone = get_backbone(conf.RES1, conf.RES2)
            privileged_model = PrivilegedDistillation(student_backbone, teacher_backbone, num_classes=1)

            print(f"Memory after loading privileged_model: {get_memory_usage():.2f} MiB")
            print(f"Model footprint: {get_memory_usage() - mem_before:.2f} MiB")

            if conf.two_stage == 0:
                privileged_model = train_privileged(privileged_model, train_loader, val_loader, trn_df, conf)
            else:
                privileged_model = train_privileged_logit_kd(privileged_model, train_loader, val_loader, trn_df, conf)
        # train_privileged returns the full wrapper; we only need the student for inference
        Models[MODE] = privileged_model.student
        
    Models["teacher"] = privileged_model.teacher

    # ============================================================
    # INFERENCE
    # ============================================================
    for side in sides:
        for MODE in MODES2:
            with track_compute(f"Infer on {side} | {MODE}"):
                preds[side, MODE, 'val'] = run_inference(Models[MODE], val_loader, side)
                preds[side, MODE, 'tst'] = run_inference(Models[MODE], test_loader, side)

    for side in sides:
        for MODE in MODES2:
            with track_compute(f"Threshold + metrics on {side} | {MODE}"):
                thresholds[side, MODE] = select_optimal_threshold(conf, preds[side, MODE, 'val'][0], preds[side, MODE, 'val'][1])  # gt_labels in 1st, probs in 2nd
                metrics[side, MODE], cis[side, MODE] = bootstrapped_metrics(preds[side, MODE, 'tst'][0], preds[side, MODE, 'tst'][1], thresholds[side, MODE], eye=side)
                # get_predictions_df(preds[side,MODE,'tst'][0], preds[side,MODE,'tst'][1], preds[side,MODE,'tst'][2], thresholds[side,MODE], test_loader, side).to_csv(f"{pref}_{MODE}_preds.csv", index=False)


    # ============================================================
    # RUN EXTERNAL EVAL
    # ============================================================
    for side in sides:
        for d in [0, 1, 2]:
            for MODE in MODES2: 
                preds[side, MODE, DSET, d] = run_inference(Models[MODE], ext_loaders[d, side], side)
                calib_set = 0
                if (side, MODE, DSET, calib_set) not in thresholds:
                    thresholds[side, MODE, DSET, calib_set] = select_optimal_threshold(conf, preds[side, MODE, DSET, calib_set][0], preds[side, MODE, DSET, calib_set][1])
                print(MODE, side, d, 'threshold (internal):', thresholds[side, MODE])
                print(MODE, side, d, 'threshold (external-calibrated):', thresholds[side, MODE, DSET, calib_set]) 
                metrics[side, MODE, DSET, d, 'uncalib'], cis[side, MODE, DSET, d, 'uncalib'] = bootstrapped_metrics(preds[side, MODE, DSET, d][0], preds[side, MODE, DSET, d][1], thresholds[side, MODE], eye=side)
                metrics[side, MODE, DSET, d], cis[side, MODE, DSET, d] = bootstrapped_metrics(preds[side, MODE, DSET, d][0], preds[side, MODE, DSET, d][1], thresholds[side, MODE, DSET, calib_set], eye=side)
    # Add this to your main script
    log_model_params("Baseline", Models[MODES[0]])
    log_model_params("Privileged Student", Models[MODES[1]] )
    log_model_params("Privileged Teacher", Models['teacher'] )
    
    exec( open('report2.py').read() )