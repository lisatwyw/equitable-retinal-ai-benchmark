import glob, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

files = sorted(glob.glob("../res/A_res392_bs32_lr1e-05_seed*_privileged_distillation_ODIR_preds.csv"))
rows = []
for f in files:
    seed = int(f.split("_seed")[1].split("_")[0])
    df = pd.read_csv(f)
    for eye in ["left", "right"]:
        sub = df[df.eye == eye]
        rows.append({
            "seed": seed, "eye": eye,
            "AUROC": roc_auc_score(sub.y_true, sub.y_prob),
            "AUPRC": average_precision_score(sub.y_true, sub.y_prob),
            "BA": balanced_accuracy_score(sub.y_true, sub.y_prob >= sub.threshold),
        })
agg = pd.DataFrame(rows).groupby("eye").agg(["mean", "std"])
print(agg.round(4))