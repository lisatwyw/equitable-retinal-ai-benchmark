# ============================================================
# PROOF 4: LINEAR PROBING 
# ============================================================
#   1. Dataloader yields 4-tuples (left, right, targets, idx);
#   2. The backbone passed to run_linear_probe is the same object
#      held by Models[...]. Freezing its parameters silently
#      freezes the trained model too. Now deep-copied.
#   3. Reported a single AUROC pooled across both eyes. Now
#      reported per eye, matching the main results table.
#   4. No calibration protocol. Now dual: threshold fixed from
#      ODIR (strict) and threshold from BRSET Canon d=0 (camera).
#   5. No persistence. Now saves JSON per run.
# ============================================================

import copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score


# ============================================================
# PROBE TRAINING (per eye results)
# ============================================================
def run_linear_probe(backbone, train_loader, val_loader, conf,
                     device=None, epochs=10, lr=1e-2, tag="probe"):
    """
    Freezes the backbone and trains a fresh linear classifier on top.
    Returns the trained model (with frozen backbone).
    """
    device = device if device is not None else DEVICE

    # Deep copy so freezing does not affect the caller's model
    bb = copy.deepcopy(backbone).to(device)

    model = SingleEyeNet(bb, num_classes=1).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    # Fresh classifier head
    model.classifier = nn.Linear(bb.embed_dim, 1).to(device)

    optimizer = optim.AdamW(model.classifier.parameters(), lr=lr)
    pos_weight = make_pos_weight(train_loader.dataset.samples, conf.task)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss, patience = np.inf, 0
    best_state = None

    print(f"[Linear Probe:{tag}] Training linear head (backbone frozen).")

    for epoch in range(1, epochs + 1):
        model.train(); train_loss = 0.0
        for batch in train_loader:
            # FIX 1: 4-tuple
            left, right, targets, _ = batch
            left, right, targets = left.to(device), right.to(device), targets.to(device)
            x = torch.cat([left, right], dim=0)
            y = torch.cat([targets, targets], dim=0)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval(); val_loss = 0.0
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                left, right, targets, _ = batch
                left, right, targets = left.to(device), right.to(device), targets.to(device)
                x = torch.cat([left, right], dim=0)
                y = torch.cat([targets, targets], dim=0)
                logits, _ = model(x)
                val_loss += criterion(logits, y).item()
                ys.append(y[:, 0].cpu().numpy())
                ps.append(torch.sigmoid(logits[:, 0]).cpu().numpy())
        val_loss /= len(val_loader)
        y_true = np.concatenate(ys); y_prob = np.concatenate(ps)
        val_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
        print(f"[Linear Probe:{tag}] Epoch {epoch:02d} | "
              f"Train={train_loss:.4f} | Val={val_loss:.4f} | Val AUROC={val_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, patience = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= conf.PATIENCE:
                print(f"[Linear Probe:{tag}] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# PROBE INFERENCE (per eye)
# ============================================================
def probe_predict_per_eye(model, loader, eye, device=None):
    """Returns y_true, y_prob for one eye."""
    device = device if device is not None else DEVICE
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, targets, _ = batch
            else:
                left, right, targets, _ = batch
                x = left if eye == "left" else right
            x = x.to(device)
            logits, _ = model(x)
            ys.append(targets[:, 0].cpu().numpy())
            ps.append(torch.sigmoid(logits[:, 0]).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


# ============================================================
# HELPERS FOR THE PROBE TABLE
# ============================================================
def _report(name, yt, yp, thr, eye):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_score, recall_score)
    pred = yp >= thr
    return {
        "name": name, "eye": eye, "threshold": float(thr),
        "AUROC": float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan"),
        "AUPRC": float(average_precision_score(yt, yp)) if len(np.unique(yt)) > 1 else float("nan"),
        "Accuracy": float(accuracy_score(yt, pred)),
        "Balanced_Accuracy": float(balanced_accuracy_score(yt, pred)),
        "F1": float(f1_score(yt, pred, zero_division=0)),
        "F1_Macro": float(f1_score(yt, pred, average='macro', zero_division=0)),
        "Precision": float(precision_score(yt, pred, zero_division=0)),
        "Recall": float(recall_score(yt, pred, zero_division=0)),
    }


# ============================================================
# ENTRY POINT
# ============================================================
def run_linear_probing():
    print("\n" + "=" * 72)
    print("PROOF 4: LINEAR PROBING")
    print("=" * 72)

    # FIX 2: deep copy so freezing does not touch the trained backbones
    baseline_backbone  = copy.deepcopy(Models["baseline"].backbone)
    privileged_backbone = copy.deepcopy(Models["privileged_distillation"].backbone)

    print("\n--- Probing Baseline backbone ---")
    baseline_probe = run_linear_probe(
        baseline_backbone, train_loader, val_loader, conf,
        epochs=10, lr=1e-2, tag="baseline")

    print("\n--- Probing Privileged student backbone ---")
    privileged_probe = run_linear_probe(
        privileged_backbone, train_loader, val_loader, conf,
        epochs=10, lr=1e-2, tag="privileged")

    # ---- Thresholds on ODIR test (strict protocol) ----
    thr_odir = {}
    for name, model in [("baseline", baseline_probe), ("privileged", privileged_probe)]:
        for eye in ["left", "right"]:
            _, p = probe_predict_per_eye(model, val_loader, eye)
            # Calibrate on the ODIR validation set, not on the test set
            yt_val, yp_val = probe_predict_per_eye(model, val_loader, eye)
            thr_odir[name, eye] = select_optimal_threshold(yt_val, yp_val)

    # ---- ODIR test metrics (per eye) ----
    print("\n" + "=" * 72)
    print("LINEAR PROBING -- ODIR TEST (per eye)")
    print("=" * 72)
    probe_metrics_odir = {}
    for name, model in [("baseline", baseline_probe), ("privileged", privileged_probe)]:
        for eye in ["left", "right"]:
            yt, yp = probe_predict_per_eye(model, test_loader, eye)
            probe_metrics_odir[name, eye] = _report(name, yt, yp, thr_odir[name, eye], eye)
            m = probe_metrics_odir[name, eye]
            print(f"  {name:10s} | {eye:5s} | AUROC={m['AUROC']:.4f} | "
                  f"AUPRC={m['AUPRC']:.4f} | BA={m['Balanced_Accuracy']:.4f}")

    # ---- External BRSET splits ----
    # Two protocols, matching the main results table.
    print("\n" + "=" * 72)
    print("LINEAR PROBING -- EXTERNAL BRSET (per eye, dual protocol)")
    print("=" * 72)
    probe_metrics_ext_strict = {}
    probe_metrics_ext_camera = {}
    n_splits = len(ext_loaders)
    for name, model in [("baseline", baseline_probe), ("privileged", privileged_probe)]:
        for eye in ["left", "right"]:
            # Strict: threshold from ODIR validation
            thr_s = thr_odir[name, eye]
            # Camera-calibrated: threshold from BRSET Canon (d=0)
            yt_c, yp_c = probe_predict_per_eye(model, ext_loaders[0], eye)
            thr_c = select_optimal_threshold(yt_c, yp_c)
            for d in range(1, n_splits):
                yt, yp = probe_predict_per_eye(model, ext_loaders[d], eye)
                probe_metrics_ext_strict[name, eye, d] = _report(name, yt, yp, thr_s, eye)
                probe_metrics_ext_camera[name, eye, d] = _report(name, yt, yp, thr_c, eye)
                ms = probe_metrics_ext_strict[name, eye, d]
                mc = probe_metrics_ext_camera[name, eye, d]
                print(f"  split {d} | {name:10s} | {eye:5s} | "
                      f"strict AUROC={ms['AUROC']:.4f} | camera AUROC={mc['AUROC']:.4f}")

    # ---- Summary delta ----
    print("\n" + "=" * 72)
    print("PROBE DELTA (privileged − baseline)")
    print("=" * 72)
    for eye in ["left", "right"]:
        d_odir = probe_metrics_odir["privileged", eye]["AUROC"] - \
                 probe_metrics_odir["baseline", eye]["AUROC"]
        print(f"  ODIR test | {eye:5s} | ΔAUROC = {d_odir:+.4f}")
        for d in range(1, n_splits):
            d_s = probe_metrics_ext_strict["privileged", eye, d]["AUROC"] - \
                  probe_metrics_ext_strict["baseline", eye, d]["AUROC"]
            d_c = probe_metrics_ext_camera["privileged", eye, d]["AUROC"] - \
                  probe_metrics_ext_camera["baseline", eye, d]["AUROC"]
            print(f"  BRSET d={d} | {eye:5s} | "
                  f"ΔAUROC strict = {d_s:+.4f} | ΔAUROC camera = {d_c:+.4f}")

    # ---- Persist ----
    def _clean(o):
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, tuple): return list(o)
        if isinstance(o, float) and (o != o): return None
        return o
    out = {
        "task": conf.task,
        "teacher_mode": conf.teacher_mode,
        "distill_loss": conf.distill_loss,
        "alpha": conf.ALPHA,
        "seed": conf.SEED,
        "odir": {f"{k[0]}|{k[1]}": v for k, v in probe_metrics_odir.items()},
        "ext_strict": {f"{k[0]}|{k[1]}|d{k[2]}": v for k, v in probe_metrics_ext_strict.items()},
        "ext_camera": {f"{k[0]}|{k[1]}|d{k[2]}": v for k, v in probe_metrics_ext_camera.items()},
    }
    path = f"{conf.pref()}_linear_probe.json"
    with open(path, "w") as f:
        json.dump(_clean(out), f, indent=2)
    print(f"\n[Linear Probing] Saved: {path}")


run_linear_probing()