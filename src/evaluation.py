import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

from typing import List
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, cohen_kappa_score, mean_absolute_error
)

BINARY_METRICS = ["AUROC", "AUPRC", "Accuracy", "Balanced_Accuracy", "Precision", "Recall", "Specificity", "F1"]
ORDINAL_METRICS = ["AUROC", "Accuracy", "Balanced_Accuracy", "Macro_F1", "Weighted_F1", "QWK", "MAE"]

def compute_metric(y_true, y_pred_or_prob, metric: str, is_ordinal: bool = False, threshold: float = 0.5):
    if is_ordinal:
        if metric == "AUROC":
            try: return roc_auc_score(y_true, y_pred_or_prob, multi_class="ovr", average="macro")
            except: return np.nan
        if metric == "Accuracy": return accuracy_score(y_true, y_pred_or_prob)
        if metric == "Balanced_Accuracy": return balanced_accuracy_score(y_true, y_pred_or_prob)
        if metric == "Macro_F1": return f1_score(y_true, y_pred_or_prob, average="macro", zero_division=0)
        if metric == "Weighted_F1": return f1_score(y_true, y_pred_or_prob, average="weighted", zero_division=0)
        if metric == "QWK": return cohen_kappa_score(y_true, y_pred_or_prob, weights="quadratic")
        if metric == "MAE": return mean_absolute_error(y_true, y_pred_or_prob)
    else:
        q = y_pred_or_prob >= threshold
        if metric == "AUROC": return roc_auc_score(y_true, y_pred_or_prob)
        if metric == "AUPRC": return average_precision_score(y_true, y_pred_or_prob)
        if metric == "Accuracy": return accuracy_score(y_true, q)
        if metric == "Balanced_Accuracy": return balanced_accuracy_score(y_true, q)
        if metric == "Precision": return precision_score(y_true, q, zero_division=0)
        if metric == "Recall": return recall_score(y_true, q, zero_division=0)
        if metric == "F1": return f1_score(y_true, q, zero_division=0)
        if metric == "Specificity":
            tn, fp, _, _ = confusion_matrix(y_true, q, labels=[0, 1]).ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return np.nan

def specificity_score(y, q):
    tn, fp, _, _ = confusion_matrix(y, q, labels=[0, 1]).ravel()
    return np.nan if (tn + fp) == 0 else tn / (tn + fp)

def calc_bin(y, p, t, m):
    y, p = np.asarray(y), np.asarray(p)
    ok = pd.notna(y) & np.isfinite(p)
    y, p = y[ok].astype(int), p[ok]
    if len(y) < 2 or len(np.unique(y)) < 2: return np.nan
    q = p >= t
    if m == "AUROC": return roc_auc_score(y, p)
    if m == "AUPRC": return average_precision_score(y, p)
    if m == "Accuracy": return accuracy_score(y, q)
    if m == "Balanced_Accuracy": return balanced_accuracy_score(y, q)
    if m == "Precision": return precision_score(y, q, zero_division=0)
    if m == "Recall": return recall_score(y, q, zero_division=0)
    if m == "Specificity": return specificity_score(y, q)
    return f1_score(y, q, zero_division=0)

def calc_ord(y, pred, prob, m):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    prob = np.asarray(prob)
    if m == "AUROC":
        try: return roc_auc_score(y, prob, multi_class="ovr", average="macro")
        except: return np.nan
    if m == "Accuracy": return accuracy_score(y, pred)
    if m == "Balanced_Accuracy": return balanced_accuracy_score(y, pred)
    if m == "Macro_F1": return f1_score(y, pred, average="macro", zero_division=0)
    if m == "Weighted_F1": return f1_score(y, pred, average="weighted", zero_division=0)
    if m == "QWK": return cohen_kappa_score(y, pred, weights="quadratic")
    return mean_absolute_error(y, pred)

def run_subgroup_bootstrap(preds_df: pd.DataFrame, resolutions: List[int], n_bootstrap: int = 100, ci_level: float = 0.95, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    alpha = 1.0 - ci_level
    r0 = resolutions[0]
    results = []
    
    sex_col, age_col = "patient_sex", "age_group"
    binary_tasks = [t for t in preds_df["task"].unique() if t != "DR_ICDR"]

    for (clf, dataset, task), g in preds_df.groupby(["classifier", "dataset", "task"]):
        is_binary = task in binary_tasks
        z = None

        # Pivot resolutions into aligned columns
        for r in resolutions:
            x = g[g.resolution == r].copy()
            if is_binary:
                x = x[["image_id", sex_col, age_col, "y", "prob", "threshold"]].rename(
                    columns={"y": f"y{r}", "prob": f"p{r}", "threshold": f"t{r}"}
                )
            else:
                pc = [c for c in x.columns if c.startswith("prob_")]
                x = x[["image_id", sex_col, age_col, "y", "pred"] + pc].rename(
                    columns={"y": f"y{r}", "pred": f"pred{r}", **{c: f"{c}_{r}" for c in pc}}
                )
            z = x if z is None else z.merge(x.drop(columns=[sex_col, age_col]), on="image_id", how="inner")

        if z is None or len(z) < 2: continue

        # Helper to compute point metrics and bootstrap CIs for a subgroup slice
        def evaluate_slice(h_df, sg_label):
            metrics = BINARY_METRICS if is_binary else ORDINAL_METRICS
            for m in metrics:
                # Point estimates
                obs = {}
                for r in resolutions:
                    if is_binary:
                        obs[r] = calc_bin(h_df[f"y{r}"], h_df[f"p{r}"], h_df[f"t{r}"].iloc[0], m)
                    else:
                        classes = sorted(int(c.split("_")[1]) for c in h_df.columns if c.startswith("prob_") and c.endswith(f"_{r}"))
                        obs[r] = calc_ord(h_df[f"y{r}"], h_df[f"pred{r}"], h_df[[f"prob_{c}_{r}" for c in classes]], m)

                # Bootstrap resampling on the subgroup slice
                bm = {r: [] for r in resolutions}
                for _ in range(n_bootstrap):
                    ix = rng.integers(len(h_df), size=len(h_df))
                    boot_sample = h_df.iloc[ix]
                    for r in resolutions:
                        if is_binary:
                            val = calc_bin(boot_sample[f"y{r}"], boot_sample[f"p{r}"], boot_sample[f"t{r}"].iloc[0], m)
                        else:
                            classes = sorted(int(c.split("_")[1]) for c in boot_sample.columns if c.startswith("prob_") and c.endswith(f"_{r}"))
                            val = calc_ord(boot_sample[f"y{r}"], boot_sample[f"pred{r}"], boot_sample[[f"prob_{c}_{r}" for c in classes]], m)
                        if np.isfinite(val): bm[r].append(val)

                ci = {
                    r: np.percentile(bm[r], [100 * (alpha / 2), 100 * (1 - alpha / 2)]) if bm[r] else [np.nan, np.nan]
                    for r in resolutions
                }

                row = {
                    "classifier": clf, "dataset": dataset, "task": task, "subgroup": sg_label,
                    "metric": m, "n": len(h_df)
                }
                for r in resolutions:
                    row[f"{r}_value"] = obs[r]
                    row[f"{r}_CI_low"] = ci[r][0]
                    row[f"{r}_CI_high"] = ci[r][1]
                results.append(row)

        # 1. Overall Population
        evaluate_slice(z, "overall")

        # 2. Subgroup Slices (Sex, Age, and Sex|Age Interaction)
        subgroup_slices = [
            *[(str(s), h) for s, h in z.groupby(sex_col, dropna=False)],
            *[(str(a), h) for a, h in z.groupby(age_col, dropna=False)],
            *[(f"{s}|{a}", h) for (s, a), h in z.groupby([sex_col, age_col], dropna=False)]
        ]

        for sg, h in subgroup_slices:
            if is_binary:
                y_sub = h[f"y{r0}"]
                # Filter out small subgroups lacking minimum positive/negative counts
                if y_sub.sum() >= 5 and (len(h) - y_sub.sum()) >= 5:
                    evaluate_slice(h, sg)
            elif len(h) >= 10:
                evaluate_slice(h, sg)

    return pd.DataFrame(results)



    