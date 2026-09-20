# ============================================================
# PRIVILEGED DISTILLATION: PAIRED-EYE TEACHER -> SINGLE-EYE STUDENT
# ============================================================
# Dual external-validation protocol:
#   (A) STRICT external   : threshold fixed from ODIR, applied to all BRSET
#   (B) CAMERA-calibrated : threshold re-selected on BRSET Canon split (d=0),
#                           applied to BRSET Nikon splits (d=1, d=2)
# ============================================================

import os, random, copy, argparse, yaml, time, logging, psutil, pickle
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, timm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, confusion_matrix, f1_score,
    recall_score, precision_score,
)
from scipy.stats import kruskal, chi2_contingency, fisher_exact

# ============================================================
# SETUP
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(device)
SEED = 41
ncores = max(1, len(os.sched_getaffinity(0)) // 2)
pin_mem = (device == "cuda")

def set_all_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_all_seeds(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
parser.add_argument("--data-dir", default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--teacher-mode", default=None)
parser.add_argument("--distill-loss", default=None)
parser.add_argument("--force-retrain", action="store_true")
args, _ = parser.parse_known_args()

if args.seed is not None:
    SEED = args.seed
    set_all_seeds(SEED)

def load_config(path):
    with open(path, "r") as f: return yaml.safe_load(f)

USERID = os.environ.get("USER", "").lower()
USERID = f"def-{USERID}-ab"

def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def specificity_score(y, q):
    tn, fp, _, _ = confusion_matrix(y, q, labels=[0, 1]).ravel()
    return np.nan if (tn + fp) == 0 else tn / (tn + fp)

# ============================================================
# CONFIG
# ============================================================
@dataclass
class Config:
    BS: int = 32
    PATIENCE: int = 10
    ALPHA: float = 0.1
    CENTROID_CROP: bool = True
    LR: float = 1e-5
    MAX_EPOCHS: int = 20
    task: str = "A"
    resolutions: List[int] = field(default_factory=lambda: [70*14, 70*14])
    res_dir: str = "../res/"
    vis_dir: str = "../vis/"
    teacher_mode: str = "full"
    noise_std: float = 0.5
    distill_loss: str = "mse"
    distill_temp: float = 4.0
    distill_conf_thresh: float = 0.5
    def checkpoint_filepath(self, method): return f"{self.pref()}_{method}.pt"
    def pref(self):
        return (
            f"{self.res_dir}"
            f"{self.task}"
            f"_res{self.resolutions[0]}"
            f"_bs{self.BS}"
            f"_lr{self.LR}"
            f"_ep{self.MAX_EPOCHS}"
            f"_pat{self.PATIENCE}"
            f"_alp{self.ALPHA}"
            f"_crop{int(self.CENTROID_CROP)}"            
            f"_tm{self.teacher_mode}"
            f"_ns{self.noise_std}"
            f"_dl{self.distill_loss}"
            f"_dt{self.distill_temp}"
            f"_dct{self.distill_conf_thresh}"
            f"_seed{self.SEED}"
        )

conf = Config()
RES1 = RES2 = conf.resolutions[0]
os.makedirs(conf.res_dir, exist_ok=True)
os.makedirs(conf.vis_dir, exist_ok=True)

logging.basicConfig(filename=f'{conf.pref()}_experiment_benchmark.log',
                    level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')

if args.config is not None:
    cfg = load_config(args.config)
    DATA_DIR = args.data_dir if args.data_dir is not None else cfg["dataset"]["data_dir"]
    metadata_file = os.path.join(DATA_DIR, cfg["dataset"]["metadata_file"])
    IM_DIR = os.path.join(DATA_DIR, cfg["dataset"]["image_subdirectory"])
    CHKPATH = cfg.get("checkpoint_path",
                      f"/project/{USERID}/HowRU/retinal/assets/retfoundgreen_statedict.pth")
    conf.MAX_EPOCHS    = cfg.get("max_epochs", conf.MAX_EPOCHS)
    conf.LR            = cfg.get("lr", conf.LR)
    conf.ALPHA         = cfg.get("alpha", conf.ALPHA)
    conf.CENTROID_CROP = cfg.get("centroid_crop", conf.CENTROID_CROP)
    conf.task          = cfg.get("TK", conf.task)
    conf.teacher_mode  = cfg.get("teacher_mode", conf.teacher_mode)
    conf.noise_std     = cfg.get("noise_std", conf.noise_std)
    conf.distill_loss  = cfg.get("distill_loss", conf.distill_loss)
    conf.distill_temp  = cfg.get("distill_temp", conf.distill_temp)
    conf.distill_conf_thresh = cfg.get("distill_conf_thresh", conf.distill_conf_thresh)
else:
    if os.path.exists("/kaggle"):
        DATA_DIR = "/kaggle/input/datasets/andrewmvd/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/"
        IM_DIR = DATA_DIR + "/Training Images"; CHKPATH = "rfg_statedict.pth"
    else:
        CHKPATH = f"/project/{USERID}/HowRU/retinal/assets/retfoundgreen_statedict.pth"
        DATA_DIR = f"/project/{USERID}/ODIR-5K/ODIR-5K/"; IM_DIR = DATA_DIR + "/trn"
    metadata_file = DATA_DIR + "data.xlsx"

if args.teacher_mode is not None: conf.teacher_mode = args.teacher_mode
if args.distill_loss is not None: conf.distill_loss = args.distill_loss
print(conf); logging.info(f"Effective config: {conf}")

# ============================================================
# DATA
# ============================================================
odir_df = pd.read_excel(metadata_file)

def reformat_and_prepare_odir(df):
    target_cols = ["N","D","G","C","A","H","M","O"]
    keyword_cols = [c for c in ["Left-Diagnostic Keywords","Right-Diagnostic Keywords"] if c in df.columns]
    out = df[["ID","Left-Fundus","Right-Fundus"]+target_cols+keyword_cols].copy()
    out = out.rename(columns={"ID":"patient_id"})
    for c in target_cols: out[c] = out[c].astype(int)
    return out, target_cols, keyword_cols

DR_KEYWORDS = ["diabetic retinopathy","proliferative diabetic retinopathy",
               "severe proliferative diabetic retinopathy",
               "mild nonproliferative retinopathy",
               "moderate non proliferative retinopathy",
               "severe nonproliferative retinopathy",
               "non proliferative retinopathy",
               "post laser photocoagulation","post retinal laser surgery"]
AMD_KEYWORDS = ["age-related macular degeneration",
                "dry age-related macular degeneration",
                "wet age-related macular degeneration"]
DR_AMBIG  = ["suspected diabetic","suspicious diabetic"]
AMD_AMBIG = ["macular hole","myopic maculopathy","idiopathic choroidal neovascularization"]

def _parse_one(kw, inc, amb):
    if pd.isna(kw): return np.nan
    k = str(kw).lower()
    tokens = [t.strip() for t in k.replace("，", ",").split(",")]
    for a in amb:
        if any(a in t for t in tokens): return np.nan
    return int(any(any(p in t for t in tokens) for p in inc))

def add_keyword_labels(df):
    if "Left-Diagnostic Keywords" not in df.columns:
        for c in ["D_left","D_right","A_left","A_right"]: df[c] = np.nan
        return df
    df["D_left"]  = df["Left-Diagnostic Keywords"].apply(lambda x: _parse_one(x, DR_KEYWORDS, DR_AMBIG))
    df["D_right"] = df["Right-Diagnostic Keywords"].apply(lambda x: _parse_one(x, DR_KEYWORDS, DR_AMBIG))
    df["A_left"]  = df["Left-Diagnostic Keywords"].apply(lambda x: _parse_one(x, AMD_KEYWORDS, AMD_AMBIG))
    df["A_right"] = df["Right-Diagnostic Keywords"].apply(lambda x: _parse_one(x, AMD_KEYWORDS, AMD_AMBIG))
    return df

def split_dataset_by_patient(df):
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    gkf = GroupKFold(n_splits=6)
    tv_idx, te_idx = next(gkf.split(df, groups=df["patient_id"]))
    tv = df.iloc[tv_idx].reset_index(drop=True); te = df.iloc[te_idx].reset_index(drop=True)
    gkf = GroupKFold(n_splits=5)
    tr_idx, va_idx = next(gkf.split(tv, groups=tv["patient_id"]))
    tr = tv.iloc[tr_idx].reset_index(drop=True); va = tv.iloc[va_idx].reset_index(drop=True)
    n = len(df)
    print(f"[SPLIT] train={len(tr)} ({len(tr)/n:.3f}) val={len(va)} ({len(va)/n:.3f}) test={len(te)} ({len(te)/n:.3f})")
    return tr, va, te

def filter_existing_images(df, image_dir):
    l = df["Left-Fundus"].apply( lambda x: os.path.exists(os.path.join(image_dir, str(x))))
    r = df["Right-Fundus"].apply(lambda x: os.path.exists(os.path.join(image_dir, str(x))))
    return df[l & r].reset_index(drop=True)

# ============================================================
# DATASET
# ============================================================
class CFPDataset(Dataset):
    def __init__(self, dataframe, target_cols, image_dir, CENTROID_CROP=True,
                 target_size=(392,392), image_col_left="Left-Fundus",
                 image_col_right="Right-Fundus", single_image=False, file_format=None):
        self.CENTROID_CROP = CENTROID_CROP
        self.target_cols = [target_cols] if isinstance(target_cols, str) else list(target_cols)
        self.image_dir = image_dir; self.target_size = target_size
        self.image_col_left = image_col_left; self.image_col_right = image_col_right
        self.single_image = single_image; self.file_format = file_format
        self.samples = dataframe.copy().reset_index(drop=True)
        for c in self.target_cols:
            self.samples[c] = pd.to_numeric(self.samples[c], errors="coerce").fillna(0).astype(np.float32)

    def _crop(self, img):
        _, h, w = img.shape
        g = img.mean(dim=0); idx = torch.nonzero(g > 0.05)
        if idx.numel() == 0: return img
        mn, _ = idx.min(dim=0); mx, _ = idx.max(dim=0)
        ymin, xmin = mn.tolist(); ymax, xmax = mx.tolist()
        d = min(ymax - ymin + 1, xmax - xmin + 1); half = d // 2
        cy, cx = (ymin+ymax)//2, (xmin+xmax)//2
        ys, ye = cy-half, cy+half; xs, xe = cx-half, cx+half
        pt, pb = max(0,-ys), max(0,ye-h); pl, pr = max(0,-xs), max(0,xe-w)
        ys, ye = max(0,ys), min(h,ye); xs, xe = max(0,xs), min(w,xe)
        crop = img[:, ys:ye, xs:xe]
        if pt or pb or pl or pr: crop = TF.pad(crop, [pl,pt,pr,pb], fill=0)
        return crop

    def _load(self, fn):
        if self.file_format is not None: fn = f"{fn}.{self.file_format}"
        path = os.path.join(self.image_dir, str(fn))
        try:
            with Image.open(path) as im: im = im.convert("RGB")
            im = TF.to_tensor(im)
            if self.CENTROID_CROP: im = self._crop(im)
        except Exception:
            im = torch.zeros(3, self.target_size[0], self.target_size[1])
        return TF.resize(im, self.target_size)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        labels = torch.tensor(row[self.target_cols].values.astype(np.float32))
        if self.single_image: return self._load(row[self.image_col_left]), labels, int(idx)
        return (self._load(row[self.image_col_left]), self._load(row[self.image_col_right]),
                labels, int(idx))

def show_batch(D, n=8, out_file=None):
    try: imgs, imgs2, labels, idxs = D
    except: imgs2 = None; imgs, labels, idxs = D
    fig, axes = plt.subplots(n, 2, figsize=(16, 4*n)); axes = axes.flatten(); c = 0
    pos = np.where(labels)[0]
    S = ([pos[0]] + list(np.arange(n-1))) if len(pos) else list(np.arange(n-1))
    for i in S:
        axes[c].imshow(np.clip(imgs[i].permute(1,2,0).cpu().numpy(), 0, 1))
        axes[c].set_title(f"Patient {idxs[i]} | outcome={labels[i].numpy()}"); axes[c].axis("off"); c += 1
        try:
            if imgs2 is not None:
                axes[c].imshow(np.clip(imgs2[i].permute(1,2,0).cpu().numpy(), 0, 1))
                axes[c].set_title(f"Patient {idxs[i]} | view2"); axes[c].axis("off"); c += 1
        except Exception: pass
        if c >= len(axes): break
    for j in range(c, len(axes)): axes[j].axis("off")
    plt.tight_layout()
    if out_file: plt.savefig(f"{conf.vis_dir}/{out_file}.png", bbox_inches="tight")
    plt.show()

# ============================================================
# BACKBONE
# ============================================================
def get_backbone():
    m = timm.create_model("vit_small_patch14_reg4_dinov2",
                          img_size=(RES1,RES2), num_classes=0, dynamic_img_size=True)
    ck = torch.load(CHKPATH, map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck
    if "pos_embed" in sd:
        pe = sd["pos_embed"]; B,N,C = pe.shape
        og = int(N**0.5); ng = m.patch_embed.grid_size[0]
        pe = pe.reshape(B,og,og,C).permute(0,3,1,2)
        pe = F.interpolate(pe, size=(ng,ng), mode="bicubic", align_corners=False)
        pe = pe.permute(0,2,3,1).reshape(B, ng*ng, C)
        sd["pos_embed"] = pe
    m.load_state_dict(sd, strict=False); m.global_pool = "avg"; return m

# ============================================================
# MODELS
# ============================================================
class SingleEyeNet(nn.Module):
    def __init__(self, bb, nc=1):
        super().__init__(); self.backbone = bb; self.classifier = nn.Linear(bb.embed_dim, nc)
    def forward(self, x): z = self.backbone(x); return self.classifier(z), z

class PrivilegedTeacher(nn.Module):
    def __init__(self, bb, nc=1, mode="full", noise_std=0.5):
        super().__init__(); self.backbone = bb; self.classifier = nn.Linear(bb.embed_dim, nc)
        self.mode = mode; self.noise_std = noise_std
    def forward(self, left, right):
        zl = self.backbone(left)
        if self.mode == "monocular": zt = zl
        elif self.mode == "masked":
            zr = self.backbone(torch.zeros_like(right)); zt = (zl + zr) / 2.0
        elif self.mode == "corrupted":
            nr = (right + self.noise_std * torch.randn_like(right)).clamp(0,1)
            zr = self.backbone(nr); zt = (zl + zr) / 2.0
        elif self.mode == "shuffled":
            perm = torch.randperm(right.size(0), device=right.device)
            zr = self.backbone(right[perm]); zt = (zl + zr) / 2.0
        elif self.mode == "full":
            zr = self.backbone(right); zt = (zl + zr) / 2.0
        else: raise ValueError(f"Unknown teacher_mode: {self.mode}")
        return self.classifier(zt), zt

class PrivilegedDistillation(nn.Module):
    def __init__(self, sb, tb, nc=1, teacher_mode="full", noise_std=0.5):
        super().__init__()
        self.student = SingleEyeNet(sb, nc)
        self.teacher = PrivilegedTeacher(tb, nc, mode=teacher_mode, noise_std=noise_std)
    def student_forward(self, x): return self.student(x)
    def teacher_forward(self, l, r): return self.teacher(l, r)

# ============================================================
# LOSS
# ============================================================
def make_pos_weight(df, task):
    p = float(df[task].values.sum()); n = float(len(df) - p)
    return torch.tensor([n / max(p, 1.0)], device=DEVICE)

def compute_distill_loss(sl, sf, tl, tf, mode="mse", temp=4.0, conf_thresh=0.5):
    tl = tl.detach(); tf = tf.detach()
    if mode == "mse": return F.mse_loss(sf, tf)
    if mode == "kl":
        s = F.log_softmax(sl/temp, dim=1); t = F.softmax(tl/temp, dim=1)
        return F.kl_div(s, t, reduction="batchmean") * (temp**2)
    if mode == "contrastive":
        zs = F.normalize(sf, dim=-1); zt = F.normalize(tf, dim=-1)
        lg = (zs @ zt.T) / 0.1; lb = torch.arange(zs.size(0), device=zs.device)
        return F.cross_entropy(lg, lb)
    if mode == "cka":
        zs = sf - sf.mean(0, keepdim=True); zt = tf - tf.mean(0, keepdim=True)
        num = torch.norm(zs.T @ zt, p='fro')**2
        den = torch.norm(zs.T @ zs, p='fro') * torch.norm(zt.T @ zt, p='fro') + 1e-12
        return 1.0 - num/den
    if mode == "gated_aux":
        with torch.no_grad():
            tp = torch.sigmoid(tl); mask = (tp > conf_thresh) | (tp < 1-conf_thresh)
        if mask.any(): return F.binary_cross_entropy_with_logits(sl[mask], tp[mask])
        return torch.tensor(0.0, device=DEVICE)
    raise ValueError(f"Unknown distill_loss: {mode}")

# ============================================================
# TRAINING
# ============================================================
def train_baseline(model, train_loader, val_loader, trn_df, conf):
    model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=conf.LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=make_pos_weight(trn_df, conf.task))
    best, pat = np.inf, 0; ckpt = conf.checkpoint_filepath("baseline")
    for ep in range(1, conf.MAX_EPOCHS+1):
        model.train(); tl = 0.0
        for l, r, t, _ in train_loader:
            l = l.to(DEVICE, non_blocking=True); r = r.to(DEVICE, non_blocking=True)
            t = t.to(DEVICE, non_blocking=True)
            x = torch.cat([l, r], 0); y = torch.cat([t, t], 0)
            lg, _ = model(x); loss = crit(lg, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item()
        tl /= len(train_loader); vl, va = evaluate_validation_single_eye(model, val_loader, crit)
        print(f"[BASELINE] Epoch {ep:02d} | Train={tl:.4f} | Val={vl:.4f} | AUROC={va:.4f}")
        if vl < best:
            best, pat = vl, 0
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "best_val_loss": best,
                        "task": conf.task, "lr": conf.LR, "max_epochs": conf.MAX_EPOCHS}, ckpt)
        else:
            pat += 1
            if pat >= conf.PATIENCE: print("[BASELINE] Early stopping."); break
    ck = torch.load(ckpt, map_location=DEVICE); model.load_state_dict(ck["state_dict"])
    return model

def train_privileged(model, train_loader, val_loader, trn_df, conf):
    model.to(DEVICE)
    opt = optim.AdamW(list(model.student.parameters()) + list(model.teacher.parameters()), lr=conf.LR)
    crit = nn.BCEWithLogitsLoss(pos_weight=make_pos_weight(trn_df, conf.task))
    best, pat = np.inf, 0
    ckpt = conf.checkpoint_filepath(f"privileged_{conf.teacher_mode}_{conf.distill_loss}")
    for ep in range(1, conf.MAX_EPOCHS+1):
        model.student.train(); model.teacher.train(); tl = 0.0
        for l, r, t, _ in train_loader:
            l = l.to(DEVICE, non_blocking=True); r = r.to(DEVICE, non_blocking=True)
            t = t.to(DEVICE, non_blocking=True)
            tlg, zt = model.teacher_forward(l, r); loss_t = crit(tlg, t)
            xs = torch.cat([l, r], 0); ys = torch.cat([t, t], 0)
            slg, zs = model.student_forward(xs)
            zts = torch.cat([zt, zt], 0); tlgs = torch.cat([tlg, tlg], 0)
            loss_c = crit(slg, ys)
            loss_d = compute_distill_loss(slg, zs, tlgs, zts,
                                          mode=conf.distill_loss, temp=conf.distill_temp,
                                          conf_thresh=conf.distill_conf_thresh)
            loss = loss_t + loss_c + conf.ALPHA * loss_d
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tl += loss.item()
        tl /= len(train_loader); vl, va = evaluate_validation_single_eye(model.student, val_loader, crit)
        print(f"[PRIV-{conf.teacher_mode}-{conf.distill_loss}] Epoch {ep:02d} | "
              f"Train={tl:.4f} | Val={vl:.4f} | AUROC={va:.4f}")
        if vl < best:
            best, pat = vl, 0
            torch.save({"student": model.student.state_dict(),
                        "teacher": model.teacher.state_dict(),
                        "teacher_mode": conf.teacher_mode,
                        "distill_loss": conf.distill_loss,
                        "alpha": conf.ALPHA, "lr": conf.LR, "max_epochs": conf.MAX_EPOCHS,
                        "task": conf.task, "epoch": ep, "best_val_loss": best}, ckpt)
        else:
            pat += 1
            if pat >= conf.PATIENCE: print("[PRIV] Early stopping."); break
    ck = torch.load(ckpt, map_location=DEVICE)
    model.student.load_state_dict(ck["student"]); model.teacher.load_state_dict(ck["teacher"])
    return model

def evaluate_validation_single_eye(model, loader, crit):
    model.eval(); ys, ps = [], []; tl = 0.0
    with torch.no_grad():
        for l, r, t, _ in loader:
            l = l.to(DEVICE, non_blocking=True); r = r.to(DEVICE, non_blocking=True)
            t = t.to(DEVICE, non_blocking=True)
            x = torch.cat([l, r], 0); y = torch.cat([t, t], 0)
            lg, _ = model(x); loss = crit(lg, y); tl += loss.item()
            ys.append(y[:, 0].cpu().numpy()); ps.append(torch.sigmoid(lg[:, 0]).cpu().numpy())
    yt = np.concatenate(ys); yp = np.concatenate(ps)
    try: a = roc_auc_score(yt, yp)
    except ValueError: a = np.nan
    return tl / len(loader), a

# ============================================================
# INFERENCE
# ============================================================
def run_inference(model, loader, eye):
    model.eval(); ys, ps, idxs = [], [], []
    with torch.no_grad():
        for b in loader:
            if len(b) == 3: x, t, i = b
            else:
                l, r, t, i = b
                x = l if eye == "left" else (r if eye == "right" else l)
            x = x.to(DEVICE, non_blocking=True)
            lg, _ = model(x)
            ys.append(t[:, 0].cpu().numpy()); ps.append(torch.sigmoid(lg[:, 0]).cpu().numpy())
            idxs.append(i.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(idxs)

def run_inference_teacher(model, loader):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for l, r, t, _ in loader:
            l = l.to(DEVICE, non_blocking=True); r = r.to(DEVICE, non_blocking=True)
            lg, _ = model.teacher_forward(l, r)
            ys.append(t[:, 0].cpu().numpy()); ps.append(torch.sigmoid(lg[:, 0]).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)

# ============================================================
# METRICS
# ============================================================
def select_optimal_threshold(y_true, y_prob):
    ts = np.linspace(0.01, 0.99, 99); bt, bs = 0.5, -np.inf
    for t in ts:
        s = f1_score(y_true, y_prob >= t, average='macro', zero_division=0)
        if s > bs: bs, bt = s, t
    return bt

def paired_bootstrap_diff(yt, pa, pb, n_boot=1000, alpha=0.05):
    rng = np.random.RandomState(SEED); n = len(yt); ds = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True); y = yt[idx]
        if len(np.unique(y)) < 2: continue
        ds.append(roc_auc_score(y, pb[idx]) - roc_auc_score(y, pa[idx]))
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [100*alpha/2, 100*(1-alpha/2)])
    p = 2 * min((ds <= 0).mean(), (ds >= 0).mean())
    return ds.mean(), (lo, hi), p

def ece(yt, yp, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1); v = 0.0
    for i in range(n_bins):
        m = (yp >= bins[i]) & (yp < bins[i+1])
        if m.sum() == 0: continue
        v += m.mean() * abs(yt[m].mean() - yp[m].mean())
    return v

def fixed_specificity_metrics(yt, yp, target_fpr=0.10):
    order = np.argsort(-yp); ys, ps = yt[order], yp[order]
    n_neg = (yt == 0).sum(); allowed = int(np.floor(target_fpr * n_neg))
    fp = 0; thr = 1.0
    for y, p in zip(ys, ps):
        if y == 0:
            fp += 1
            if fp > allowed: thr = p; break
    pred = yp >= thr
    return {"BA@FPR0.10": balanced_accuracy_score(yt, pred), "Threshold@FPR0.10": thr}

def bootstrapped_metrics(yt, yp, thr, eye="unknown", n_bootstraps=1000, alpha=0.05):
    pred = yp >= thr
    M = {"Eye": eye, "AUROC": roc_auc_score(yt, yp),
         "AUPRC": average_precision_score(yt, yp),
         "Accuracy": accuracy_score(yt, pred),
         "Balanced_Accuracy": balanced_accuracy_score(yt, pred),
         "F1": f1_score(yt, pred, zero_division=0),
         "F1_Macro": f1_score(yt, pred, average='macro', zero_division=0),
         "Precision": precision_score(yt, pred, zero_division=0),
         "Recall": recall_score(yt, pred, zero_division=0),
         "Specificity": specificity_score(yt, pred),
         "ECE": ece(yt, yp)}
    M.update(fixed_specificity_metrics(yt, yp, target_fpr=0.10))
    tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    M["FPR"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    rng = np.random.RandomState(SEED)
    B = {k: [] for k in M if k != "Eye"}; n = len(yt)
    for _ in range(n_bootstraps):
        idx = rng.choice(n, n, replace=True)
        a = yt[idx]; b = yp[idx]; q = b >= thr
        if len(np.unique(a)) < 2: continue
        for k, f in [("AUROC", lambda: roc_auc_score(a, b)),
                     ("AUPRC", lambda: average_precision_score(a, b)),
                     ("Accuracy", lambda: accuracy_score(a, q)),
                     ("Balanced_Accuracy", lambda: balanced_accuracy_score(a, q)),
                     ("F1", lambda: f1_score(a, q, zero_division=0)),
                     ("F1_Macro", lambda: f1_score(a, q, average='macro', zero_division=0)),
                     ("Precision", lambda: precision_score(a, q, zero_division=0)),
                     ("Recall", lambda: recall_score(a, q, zero_division=0)),
                     ("Specificity", lambda: specificity_score(a, q)),
                     ("ECE", lambda: ece(a, b)),
                     ("BA@FPR0.10", lambda: fixed_specificity_metrics(a, b)["BA@FPR0.10"])]:
            try: B[k].append(f())
            except Exception: pass
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(a, q).ravel()
        B["FPR"].append(fp_b/(fp_b+tn_b) if (fp_b+tn_b) > 0 else 0.0)
    C = {}
    for k, v in B.items():
        C[k] = (np.percentile(v, 100*alpha/2), np.percentile(v, 100*(1-alpha/2))) if v else (np.nan, np.nan)
    return M, C

def format_metric(name, value, ci):
    return f"{name:18s} {value:.4f}  [{ci[0]:.4f} – {ci[1]:.4f}]"

def stratified_keyword_error_analysis(pl, pr, df_kw, task):
    yl_t, yl_p, idx_l = pl; yr_t, yr_p, idx_r = pr
    df_l = df_kw.iloc[idx_l].reset_index(drop=True).assign(p_left=yl_p, y_true=yl_t)
    df_r = df_kw.iloc[idx_r].reset_index(drop=True)
    merged = df_l[["patient_id", task, f"{task}_left", f"{task}_right", "p_left", "y_true"]].copy()
    merged["p_right"] = df_r["p_right"].values if "p_right" in df_r else yr_p
    mask = merged[f"{task}_left"].notna() & merged[f"{task}_right"].notna()
    m = merged[mask].copy()
    strata = {"both affected": (m[f"{task}_left"]==1)&(m[f"{task}_right"]==1),
              "left only":     (m[f"{task}_left"]==1)&(m[f"{task}_right"]==0),
              "right only":    (m[f"{task}_left"]==0)&(m[f"{task}_right"]==1),
              "neither":       (m[f"{task}_left"]==0)&(m[f"{task}_right"]==0)}
    print(f"\n[KEYWORD ERROR ANALYSIS] Task {task} | both-eyes-labelled: {len(m)}")
    for name, sel in strata.items():
        if sel.sum() == 0: continue
        sub = m[sel]; yp = sub["y_true"].values
        pl_, pr_ = sub["p_left"].values, sub["p_right"].values
        agree = np.mean((pl_ >= 0.5) == (pr_ >= 0.5)); bias = np.mean(np.abs(pl_ - pr_))
        al = accuracy_score(yp, (pl_ >= 0.5).astype(int))
        ar = accuracy_score(yp, (pr_ >= 0.5).astype(int))
        print(f"  [{name:16s}] n={sel.sum():4d} | acc(L)={al:.3f} acc(R)={ar:.3f} | "
              f"L/R agree={agree:.3f} | mean|Δp|={bias:.3f}")

# ============================================================
# BRSET EXTERNAL SPLITS
# ============================================================
def define_splits(cfp_df):
    cfp_df["age_group"] = pd.cut(cfp_df['patient_age'],
        bins=[-np.inf,39,49,59,69,79,np.inf],
        labels=["<40","40-49","50-59","60-69","70-79","80+"])
    quality_vars = ["focus","illumination","image_field","artifacts"]
    canon = set(cfp_df.loc[cfp_df.camera == "Canon CR", "patient_id"])
    nikon = cfp_df[(cfp_df.camera == "NIKON NF5050") & ~cfp_df.patient_id.isin(canon)].copy()
    sec_p = set(nikon[nikon.image_field == 2].patient_id)
    sec_df = nikon[nikon.patient_id.isin(sec_p)]
    ext_df_ = nikon[~nikon.patient_id.isin(sec_p)]
    canon_df = cfp_df[cfp_df.patient_id.isin(canon)]
    print("overlap (nikon & canon):", len(set(nikon["patient_id"]) & set(canon_df["patient_id"])))
    print("overlap (secondary & external):", len(set(sec_df["patient_id"]) & set(ext_df_["patient_id"])))
    df = pd.concat([canon_df.assign(dataset="Canon"),
                    sec_df.assign(dataset="Nikon-abnormal"),
                    ext_df_.assign(dataset="Nikon-normal")], ignore_index=True)
    print(df.groupby("camera")[quality_vars].agg(["mean","std","median"]).round(3))
    try:
        print(ext_df_.groupby("resolution")[quality_vars].agg(["count","mean","std","median"]).round(3))
        for v in quality_vars:
            g = [x[v].dropna() for _, x in nikon.groupby("resolution")]
            try: print(v, kruskal(*g))
            except Exception: pass
    except Exception: pass
    print(df[quality_vars].eq(2).groupby(df["camera"]).mean().mul(100).round(2))
    rows = []
    for v in quality_vars:
        t = pd.crosstab(df.camera, df[v]).reindex(columns=[1,2], fill_value=0)
        a, b, c, d = t.loc["NIKON NF5050",2], t.loc["NIKON NF5050",1], t.loc["Canon CR",2], t.loc["Canon CR",1]
        or_, pf = fisher_exact([[a,b],[c,d]])
        chi2, p, _, _ = chi2_contingency(t); V = np.sqrt(chi2 / t.to_numpy().sum())
        rows.append([v.replace("_"," ").title(), f"{100*a/(a+b):.2f}", f"{100*c/(c+d):.2f}",
                     f"{or_:.2f}", f"{pf:.4f}", f"{V:.3f}"])
    res = pd.DataFrame(rows, columns=["Quality","Nikon abnormal (%)","Canon abnormal (%)","OR","p","Cramer's V"])
    print('-'*10); print('Results in latex:', res.to_latex(index=False, escape=False))
    d0 = CFPDataset(canon_df, target_cols=conf.task, image_dir=CFP_IMAGE_DIR, image_col_left="image_id",
                    image_col_right=None, single_image=True, file_format="jpg", CENTROID_CROP=conf.CENTROID_CROP)
    d1 = CFPDataset(sec_df,   target_cols=conf.task, image_dir=CFP_IMAGE_DIR, image_col_left="image_id",
                    image_col_right=None, single_image=True, file_format="jpg", CENTROID_CROP=conf.CENTROID_CROP)
    d2 = CFPDataset(ext_df_,  target_cols=conf.task, image_dir=CFP_IMAGE_DIR, image_col_left="image_id",
                    image_col_right=None, single_image=True, file_format="jpg", CENTROID_CROP=conf.CENTROID_CROP)
    L = {0: DataLoader(d0, batch_size=conf.BS, shuffle=True, num_workers=ncores, pin_memory=pin_mem),
         1: DataLoader(d1, batch_size=conf.BS, shuffle=True, num_workers=ncores, pin_memory=pin_mem),
         2: DataLoader(d2, batch_size=conf.BS, shuffle=True, num_workers=ncores, pin_memory=pin_mem)}
    return canon_df, sec_df, ext_df_, L

# ============================================================
# LOGGING
# ============================================================
@contextmanager
def track_compute(name, log_to_file=True):
    p = psutil.Process(os.getpid()); mb = p.memory_info().rss
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gb = torch.cuda.memory_allocated()/(1024**2); grb = torch.cuda.memory_reserved()/(1024**2)
    else: gb = grb = 0
    t0 = time.perf_counter()
    try: yield
    finally:
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ma = p.memory_info().rss
        if torch.cuda.is_available():
            ga = torch.cuda.memory_allocated()/(1024**2); gra = torch.cuda.memory_reserved()/(1024**2)
        else: ga = gra = 0
        el = time.perf_counter() - t0
        msg = (f"[{name}] Time: {el:.2f}s | CPU: {ma/(1024**2):.0f} MiB "
               f"(Δ {(ma-mb)/(1024**2):+.0f}) | GPU alloc: {ga:.0f} MiB (Δ {ga-gb:+.0f}) | "
               f"GPU resv: {gra:.0f} MiB (Δ {gra-grb:+.0f})")
        print(msg)
        if log_to_file: logging.info(msg)

def log_model_params(name, model):
    count = sum(p.numel() for p in model.parameters())
    print(f"[{name}] Parameters: {count:,} | Est. Size: {(count*4)/(1024**2):.2f} MiB")
    logging.info(f"[{name}] Parameters: {count:,}")
    for n, m in model.named_children():
        c = sum(p.numel() for p in m.parameters())
        print(f"  {n}: {c:,}"); logging.info(f"  {n}: {c:,}")
    return count

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    Models, preds, thresholds, cis, metrics = {}, {}, {}, {}, {}
    sides = ['left', 'right']
    MODES = ['baseline', 'privileged_distillation']

    # ---- BRSET external set ----
    try:
        CFP_DATA_ROOT = f'/project/{USERID}/physionet.org/files/brazilian-ophthalmological/1.0.2'
        CFP_IMAGE_DIR = os.path.join(CFP_DATA_ROOT, "fundus_photos")
        ext_df = pd.read_csv(os.path.join(CFP_DATA_ROOT, "label_brset.csv"))
    except Exception:
        conf.res_dir = './'; conf.vis_dir = './'
        CFP_DATA_ROOT = '/kaggle/input/datasets/tanzinabdul/fundus-patientwise-split/dataset_split/train/'
        CFP_IMAGE_DIR = '/kaggle/input/datasets/tanzinabdul/fundus-patientwise-split/dataset_split/train/images'
        ext_df = pd.read_csv('/kaggle/input/datasets/tanzinabdul/fundus-patientwise-split/dataset_split/train/train.csv')
    ext_df['D'] = ext_df.diabetic_retinopathy
    ext_df['A'] = ext_df.amd
    ext_df['M'] = ext_df.myopic_fundus

    with track_compute("Reading external dataset + generating figure of samples"):
        DSET = 'brset'
        ext_df_cal, ext_df_sub, ext_df_sub2, ext_loaders = define_splits(ext_df)
        show_batch(next(iter(ext_loaders[0])), 8, f'{conf.task}_{DSET}')

    # ---- ODIR development set ----
    df, target_cols, keyword_cols = reformat_and_prepare_odir(odir_df)
    df = add_keyword_labels(df)
    trn_df, val_df, tst_df = split_dataset_by_patient(df)
    trn_df = filter_existing_images(trn_df, IM_DIR)
    val_df = filter_existing_images(val_df, IM_DIR)
    tst_df = filter_existing_images(tst_df, IM_DIR)
    print(f"Train patients: {len(trn_df)} | Val: {len(val_df)} | Test: {len(tst_df)}")

    with track_compute("Reading DEV dataset + generating figure of samples"):
        train_loader = DataLoader(CFPDataset(trn_df, conf.task, IM_DIR, CENTROID_CROP=conf.CENTROID_CROP),
                                  batch_size=conf.BS, shuffle=True, num_workers=ncores, pin_memory=pin_mem)
        val_loader   = DataLoader(CFPDataset(val_df, conf.task, IM_DIR, CENTROID_CROP=conf.CENTROID_CROP),
                                  batch_size=conf.BS, shuffle=False, num_workers=ncores, pin_memory=pin_mem)
        test_loader  = DataLoader(CFPDataset(tst_df, conf.task, IM_DIR, CENTROID_CROP=conf.CENTROID_CROP),
                                  batch_size=conf.BS, shuffle=False, num_workers=ncores, pin_memory=pin_mem)
        show_batch(next(iter(test_loader)), 8, f'{conf.task}_ODIR_tst')

    # ---- BASELINE ----
    MODE = "baseline"; ckpt_path = conf.checkpoint_filepath(MODE)
    Models[MODE] = SingleEyeNet(get_backbone(), 1); reloaded = False
    if os.path.exists(ckpt_path) and not args.force_retrain:
        try:
            ck = torch.load(ckpt_path, map_location=DEVICE)
            Models[MODE].load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
            print(f"[RELOAD] Baseline from {ckpt_path} (epoch={ck.get('epoch')})"); reloaded = True
        except Exception as e: print(f"[SKIP] Baseline load failed: {e}")
    if not reloaded:
        with track_compute(f"Training {MODE}"):
            Models[MODE] = train_baseline(Models[MODE], train_loader, val_loader, trn_df, conf)

    # ---- PRIVILEGED ----
    MODE = "privileged_distillation"
    ckpt_path = conf.checkpoint_filepath(f"privileged_{conf.teacher_mode}_{conf.distill_loss}")
    reloaded = False
    if os.path.exists(ckpt_path) and not args.force_retrain:
        try:
            ck = torch.load(ckpt_path, map_location=DEVICE)
            if (ck.get("teacher_mode") == conf.teacher_mode
                    and ck.get("distill_loss") == conf.distill_loss
                    and ck.get("task") == conf.task):
                sb = get_backbone(); tb = get_backbone()
                priv = PrivilegedDistillation(sb, tb, 1, teacher_mode=conf.teacher_mode,
                                              noise_std=conf.noise_std)
                priv.student.load_state_dict(ck["student"]); priv.teacher.load_state_dict(ck["teacher"])
                Models[MODE] = priv.student; Models["teacher"] = priv.teacher
                print(f"[RELOAD] Privileged from {ckpt_path} (epoch={ck.get('epoch')}, "
                      f"teacher={ck.get('teacher_mode')}, loss={ck.get('distill_loss')})")
                reloaded = True
            else: print("[SKIP] Checkpoint metadata mismatch.")
        except Exception as e: print(f"[SKIP] Privileged load failed: {e}")
    if not reloaded:
        with track_compute(f"Training {MODE} | teacher={conf.teacher_mode} | loss={conf.distill_loss}"):
            sb = get_backbone(); tb = get_backbone()
            priv = PrivilegedDistillation(sb, tb, 1, teacher_mode=conf.teacher_mode,
                                          noise_std=conf.noise_std)
            priv = train_privileged(priv, train_loader, val_loader, trn_df, conf)
        Models[MODE] = priv.student; Models["teacher"] = priv.teacher

    # ---- ODIR inference ----
    for side in sides:
        for MODE in MODES:
            with track_compute(f"Infer ODIR test | {side} | {MODE}"):
                preds[side, MODE] = run_inference(Models[MODE], test_loader, side)
    with track_compute("Infer teacher (bilateral, ODIR)"):
        preds["teacher"] = run_inference_teacher(Models["teacher"], test_loader)

    # ---- ODIR thresholds + metrics ----
    for side in sides:
        for MODE in MODES:
            thresholds[side, MODE] = select_optimal_threshold(preds[side, MODE][0], preds[side, MODE][1])
    for side in sides:
        for MODE in MODES:
            metrics[side, MODE], cis[side, MODE] = bootstrapped_metrics(
                preds[side, MODE][0], preds[side, MODE][1], thresholds[side, MODE], eye=side)
    thr_t = select_optimal_threshold(preds["teacher"][0], preds["teacher"][1])
    metrics["teacher"], cis["teacher"] = bootstrapped_metrics(
        preds["teacher"][0], preds["teacher"][1], thr_t, eye="bilateral")

    # ---- Save ODIR predictions ----
    for MODE in MODES:
        b1 = get_predictions_df(preds['left', MODE][0], preds['left', MODE][1], preds['left', MODE][2],
                                thresholds['left', MODE], test_loader, 'left') if False else None
    for MODE in MODES:
        b1 = pd.DataFrame({
            "patient_id": test_loader.dataset.samples.iloc[preds['left', MODE][2]]["patient_id"].values,
            "y_true": preds['left', MODE][0], "y_prob": preds['left', MODE][1],
            "eye": "left", "threshold": thresholds['left', MODE], "dataset": "ODIR"})
        b2 = pd.DataFrame({
            "patient_id": test_loader.dataset.samples.iloc[preds['right', MODE][2]]["patient_id"].values,
            "y_true": preds['right', MODE][0], "y_prob": preds['right', MODE][1],
            "eye": "right", "threshold": thresholds['right', MODE], "dataset": "ODIR"})
        pd.concat([b1, b2], ignore_index=True).to_csv(f"{conf.pref()}_{MODE}_ODIR_preds.csv", index=False)

    # ============================================================
    # EXTERNAL BRSET EVALUATION -- DUAL PROTOCOL
    # ============================================================
    # Inference on all three BRSET splits
    for d in range(len(ext_loaders)):
        for side in sides:
            for MODE in MODES:
                key = (side, MODE, DSET, d)
                if key not in preds:
                    preds[key] = run_inference(Models[MODE], ext_loaders[d], side)

    # Protocol 1: STRICT -- threshold fixed from ODIR test set
    thresholds_strict = {}
    metrics_strict, cis_strict = {}, {}
    for side in sides:
        for MODE in MODES:
            thresholds_strict[side, MODE] = thresholds[side, MODE]   # from ODIR test
    for d in range(len(ext_loaders)):
        for side in sides:
            for MODE in MODES:
                yt, yp, _ = preds[side, MODE, DSET, d]
                metrics_strict[side, MODE, d], cis_strict[side, MODE, d] = bootstrapped_metrics(
                    yt, yp, thresholds_strict[side, MODE], eye=side)

    # Protocol 2: CAMERA-CALIBRATED -- threshold from BRSET Canon (d=0)
    thresholds_camera = {}
    metrics_camera, cis_camera = {}, {}
    calib_split = 0
    for side in sides:
        for MODE in MODES:
            yt_c, yp_c, _ = preds[side, MODE, DSET, calib_split]
            thresholds_camera[side, MODE] = select_optimal_threshold(yt_c, yp_c)
    for d in range(len(ext_loaders)):
        for side in sides:
            for MODE in MODES:
                yt, yp, _ = preds[side, MODE, DSET, d]
                metrics_camera[side, MODE, d], cis_camera[side, MODE, d] = bootstrapped_metrics(
                    yt, yp, thresholds_camera[side, MODE], eye=side)

    # ---- Save BRSET predictions (both protocols) ----
    for MODE in MODES:
        for d in range(len(ext_loaders)):
            for side in sides:
                yt, yp, idx = preds[side, MODE, DSET, d]
                dfp = pd.DataFrame({
                    "patient_id": ext_loaders[d].dataset.samples.iloc[idx]["patient_id"].values,
                    "y_true": yt, "y_prob": yp, "eye": side, "split": d,
                    "thr_strict": thresholds_strict[side, MODE],
                    "thr_camera": thresholds_camera[side, MODE],
                    "dataset": "BRSET"})
                dfp.to_csv(f"{conf.pref()}_{MODE}_BRSET_split{d}_{side}_preds.csv", index=False)

    # ============================================================
    # PRINT RESULTS
    # ============================================================
    print("\n" + "="*100)
    print(f"ODIR TEST RESULTS | task={conf.task} | teacher_mode={conf.teacher_mode} | "
          f"distill_loss={conf.distill_loss} | alpha={conf.ALPHA}")
    print("="*100)
    for MODE in MODES + ["teacher"]:
        if MODE == "teacher":
            print(f"\n{MODE} | bilateral | threshold={thr_t:.3f}")
            for k in ["AUROC","AUPRC","Accuracy","Balanced_Accuracy","F1","F1_Macro",
                      "Precision","Recall","Specificity","FPR","ECE","BA@FPR0.10"]:
                if k in metrics["teacher"]:
                    print(format_metric(k, metrics["teacher"][k], cis["teacher"][k]))
            continue
        for side in sides:
            print(f"\n{MODE} | {side} | threshold={thresholds[side, MODE]:.3f}")
            for k in ["AUROC","AUPRC","Accuracy","Balanced_Accuracy","F1","F1_Macro",
                      "Precision","Recall","Specificity","FPR","ECE","BA@FPR0.10"]:
                if k in metrics[side, MODE]:
                    print(format_metric(k, metrics[side, MODE][k], cis[side, MODE][k]))

    print("\n" + "="*100)
    print("PAIRED BOOTSTRAP: ΔAUROC (privileged − baseline) on ODIR test")
    print("="*100)
    for side in sides:
        d, ci, p = paired_bootstrap_diff(
            preds[side, "baseline"][0], preds[side, "baseline"][1],
            preds[side, "privileged_distillation"][1])
        print(f"{side:6s}: ΔAUROC={d:+.4f}  95% CI=({ci[0]:+.4f}, {ci[1]:+.4f})  p={p:.4f}")

    # ---- STRICT EXTERNAL ----
    print("\n" + "="*100)
    print(f"[PROTOCOL 1] STRICT EXTERNAL BRSET | threshold fixed from ODIR test")
    print("="*100)
    for MODE in MODES:
        for side in sides:
            print(f"\n{MODE} | {side} | thr_strict={thresholds_strict[side, MODE]:.3f}")
            for d in range(len(ext_loaders)):
                split_name = ["Canon (calib-only)", "Nikon-abnormal", "Nikon-normal"][d]
                print(f"  -- split {d} ({split_name})")
                for k in ["AUROC","AUPRC","Accuracy","Balanced_Accuracy","F1","F1_Macro",
                          "Precision","Recall","Specificity","FPR","ECE","BA@FPR0.10"]:
                    if k in metrics_strict[side, MODE, d]:
                        print("   " + format_metric(k, metrics_strict[side, MODE, d][k],
                                                    cis_strict[side, MODE, d][k]))

    # ---- CAMERA-CALIBRATED ----
    print("\n" + "="*100)
    print(f"[PROTOCOL 2] CAMERA-CALIBRATED BRSET | threshold from BRSET Canon split")
    print("="*100)
    for MODE in MODES:
        for side in sides:
            print(f"\n{MODE} | {side} | thr_camera={thresholds_camera[side, MODE]:.3f}")
            for d in range(len(ext_loaders)):
                split_name = ["Canon (calib-only)", "Nikon-abnormal", "Nikon-normal"][d]
                print(f"  -- split {d} ({split_name})")
                for k in ["AUROC","AUPRC","Accuracy","Balanced_Accuracy","F1","F1_Macro",
                          "Precision","Recall","Specificity","FPR","ECE","BA@FPR0.10"]:
                    if k in metrics_camera[side, MODE, d]:
                        print("   " + format_metric(k, metrics_camera[side, MODE, d][k],
                                                    cis_camera[side, MODE, d][k]))

    # ---- Delta between protocols on Nikon splits ----
    print("\n" + "="*100)
    print("PROTOCOL COMPARISON on Nikon splits (camera-calibrated − strict)")
    print("="*100)
    for MODE in MODES:
        for side in sides:
            for d in [1, 2]:
                for k in ["AUROC", "AUPRC", "Balanced_Accuracy", "BA@FPR0.10", "ECE"]:
                    if k in metrics_strict[side, MODE, d] and k in metrics_camera[side, MODE, d]:
                        dv = metrics_camera[side, MODE, d][k] - metrics_strict[side, MODE, d][k]
                        print(f"{MODE[:12]:12s} | {side:5s} | split {d} | {k:18s}: {dv:+.4f}")

    # ---- Keyword stratified error analysis on ODIR test ----
    with track_compute("Keyword stratified error analysis"):
        for task in ["D", "A"]:
            for M in MODES:
                stratified_keyword_error_analysis(preds['left', M], preds['right', M], tst_df, task)

    # ---- Parameter counts ----
    log_model_params("Baseline",          Models[MODES[0]])
    log_model_params("Privileged Student", Models[MODES[1]])
    log_model_params("Privileged Teacher", Models['teacher'])

    # ---- Downstream analyses ----
    for extra in ("report.py", "cka.py", "linear_prob.py"):
        if os.path.exists(extra):
            exec(open(extra).read())