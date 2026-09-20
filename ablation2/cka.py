# ============================================================
# CKA ANALYSIS WITH PER-EYE COMPARISON (no feature duplication)
# ============================================================
# Addresses the limitation in the original script where teacher
# features were duplicated as torch.cat([z_t, z_t], 0), which
# inflates/deflates CKA depending on normalization.
#
# Instead, we compare each eye's student features against the
# teacher's bilateral features directly, and report CKA per eye.
#
# Reviewer concerns addressed:
#   §3.5 (5)   AMD CKA "inconclusive" → rank and condition diagnostics
#   §3.5 (5)   only ΔCKA reported → absolute CKA reported per eye and task
#   §A.2 (1)   linear kernel called cosine without L2 normalization
#   §A.2 (2)   O(n^2) Gram matrices → dual formulation, O(nd^2)
#   §3.2       laterality asymmetry → per-eye CKA comparison
# ============================================================

import os
import json
import numpy as np
import torch

# DEVICE, Models, conf, val_loader, SEED are inherited from the training script.
# This block is designed to be exec()'d at the end of the main script.


# ============================================================
# CORE: NUMERICALLY STABLE LINEAR CKA
# ============================================================
def cka_similarity(X, Y, normalize=True):
    """
    Linear CKA via the dual formulation tr(K_c L_c) = ||X_c^T Y_c||_F^2.

    If normalize=True, rows of X and Y are L2-normalized before centering,
    so the linear kernel equals cosine similarity between samples.

    Returns a dict with CKA, rank, condition number, and shapes.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]

    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    num = np.linalg.norm(XtY, ord='fro') ** 2
    den = np.linalg.norm(XtX, ord='fro') * np.linalg.norm(YtY, ord='fro')

    rank_X = int(np.linalg.matrix_rank(X))
    rank_Y = int(np.linalg.matrix_rank(Y))
    cond_X = float(np.linalg.cond(XtX)) if XtX.size else np.nan
    cond_Y = float(np.linalg.cond(YtY)) if YtY.size else np.nan

    if den == 0:
        print(f"[CKA] Degenerate case: rank_X={rank_X}, rank_Y={rank_Y}, "
              f"cond_X={cond_X:.2e}, cond_Y={cond_Y:.2e}, n={n}")
        return {"CKA": np.nan, "rank_X": rank_X, "rank_Y": rank_Y,
                "cond_X": cond_X, "cond_Y": cond_Y,
                "n": n, "d_X": X.shape[1], "d_Y": Y.shape[1]}

    return {"CKA": num / den, "rank_X": rank_X, "rank_Y": rank_Y,
            "cond_X": cond_X, "cond_Y": cond_Y,
            "n": n, "d_X": X.shape[1], "d_Y": Y.shape[1]}


# ============================================================
# FEATURE COLLECTION (per eye, no duplication)
# ============================================================
def collect_per_eye_features(n_batches=20):
    """
    Collects feature matrices aligned per eye:

        X_baseline_left   : baseline on left eye
        X_baseline_right  : baseline on right eye
        X_priv_left       : privileged student on left eye
        X_priv_right      : privileged student on right eye
        X_teacher         : privileged teacher on the bilateral pair

    Rows correspond 1:1 to the same patient in every matrix, so CKA
    between student and teacher is computed on the same patients
    without duplication.
    """
    teacher_model      = Models["teacher"]
    baseline_model     = Models["baseline"]
    privileged_student = Models["privileged_distillation"]

    for m in (teacher_model, baseline_model, privileged_student):
        m.eval()

    feat = {
        "baseline_left":  [], "baseline_right": [],
        "priv_left":      [], "priv_right":     [],
        "teacher":        [],
    }

    with torch.no_grad():
        for i, (left, right, targets, _) in enumerate(val_loader):
            if i >= n_batches:
                break
            left  = left.to(DEVICE, non_blocking=True)
            right = right.to(DEVICE, non_blocking=True)

            _, z_b_left  = baseline_model(left)
            _, z_b_right = baseline_model(right)
            feat["baseline_left"].append(z_b_left.cpu().numpy())
            feat["baseline_right"].append(z_b_right.cpu().numpy())

            _, z_p_left  = privileged_student(left)
            _, z_p_right = privileged_student(right)
            feat["priv_left"].append(z_p_left.cpu().numpy())
            feat["priv_right"].append(z_p_right.cpu().numpy())

            _, z_t = teacher_model(left, right)
            feat["teacher"].append(z_t.cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in feat.items()}


# ============================================================
# REPORTING
# ============================================================
def _fmt(r):
    if np.isnan(r["CKA"]):
        return (f"CKA=nan (rank_X={r['rank_X']}, rank_T={r['rank_Y']}, "
                f"cond_X={r['cond_X']:.2e}, cond_T={r['cond_Y']:.2e})")
    return (f"CKA={r['CKA']:.4f}  rank_X={r['rank_X']}  rank_T={r['rank_Y']}  "
            f"cond_X={r['cond_X']:.2e}  cond_T={r['cond_Y']:.2e}")


def run_cka_per_eye(n_batches=20):
    print("\n" + "=" * 76)
    print("CKA ANALYSIS (per eye, no teacher-feature duplication)")
    print("=" * 76)

    try:
        F = collect_per_eye_features(n_batches=n_batches)
    except Exception as e:
        print(f"[CKA] Feature collection failed: {e}")
        return {}

    results = {}

    # ---- Baseline vs teacher, per eye ----
    r_bl = cka_similarity(F["baseline_left"],  F["teacher"], normalize=True)
    r_br = cka_similarity(F["baseline_right"], F["teacher"], normalize=True)
    # ---- Privileged student vs teacher, per eye ----
    r_pl = cka_similarity(F["priv_left"],  F["teacher"], normalize=True)
    r_pr = cka_similarity(F["priv_right"], F["teacher"], normalize=True)

    print(f"\n  Baseline   | left  eye vs teacher : {_fmt(r_bl)}")
    print(f"  Baseline   | right eye vs teacher : {_fmt(r_br)}")
    print(f"  Privileged | left  eye vs teacher : {_fmt(r_pl)}")
    print(f"  Privileged | right eye vs teacher : {_fmt(r_pr)}")

    # ---- ΔCKA per eye ----
    if not np.isnan(r_pl["CKA"]) and not np.isnan(r_bl["CKA"]):
        print(f"\n  ΔCKA left  (privileged − baseline) = {r_pl['CKA'] - r_bl['CKA']:+.4f}")
    if not np.isnan(r_pr["CKA"]) and not np.isnan(r_br["CKA"]):
        print(f"  ΔCKA right (privileged − baseline) = {r_pr['CKA'] - r_br['CKA']:+.4f}")

    # ---- Laterality asymmetry check ----
    if not np.isnan(r_pl["CKA"]) and not np.isnan(r_pr["CKA"]):
        print(f"\n  Laterality (privileged): |CKA_left − CKA_right| = "
              f"{abs(r_pl['CKA'] - r_pr['CKA']):.4f}")

    results = {
        "task": conf.task,
        "teacher_mode": conf.teacher_mode,
        "distill_loss": conf.distill_loss,
        "alpha": conf.ALPHA,
        "seed": conf.SEED,
        "n_samples": int(r_bl["n"]),
        "baseline_left":   r_bl,
        "baseline_right":  r_br,
        "privileged_left": r_pl,
        "privileged_right": r_pr,
        "delta_left":  None if np.isnan(r_pl["CKA"]) or np.isnan(r_bl["CKA"])
                       else float(r_pl["CKA"] - r_bl["CKA"]),
        "delta_right": None if np.isnan(r_pr["CKA"]) or np.isnan(r_br["CKA"])
                       else float(r_pr["CKA"] - r_br["CKA"]),
    }

    # ---- Persist for reproducibility ----
    out_path = f"{conf.pref()}_cka_per_eye.json"
    def _clean(d):
        if isinstance(d, dict):
            return {k: _clean(v) for k, v in d.items()}
        if isinstance(d, float) and np.isnan(d):
            return None
        return d
    with open(out_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"\n[CKA] Saved: {out_path}")
    return results


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    # Requires Models, val_loader, conf, SEED, DEVICE from the training script.
    run_cka_per_eye(n_batches=20)