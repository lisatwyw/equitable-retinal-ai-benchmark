# ============================================================
# report.py -- Dual-protocol transposed LaTeX table
# ============================================================
# Reads:
#   metrics[side, mode]           -> ODIR internal test
#   cis[side, mode]
#   metrics_strict[side, mode, d] -> BRSET strict (threshold from ODIR)
#   cis_strict[side, mode, d]
#   metrics_camera[side, mode, d] -> BRSET camera-calibrated (threshold from Canon)
#   cis_camera[side, mode, d]
#
# Split convention:
#   d = 0 : Canon CR            (calibration only, not a held-out result)
#   d = 1 : Nikon NF5050        (abnormal image field)
#   d = 2 : Nikon NF5050        (normal image field)
#
#
# exec( open('report.py').read() ) 
#
# ============================================================

from pathlib import Path


METRICS = [
    "AUROC", "AUPRC", "Accuracy", "Balanced_Accuracy",
    "F1", "F1_Macro", "Precision", "Recall", "Specificity",
    "FPR", #"ECE", "BA@FPR0.10",
]

METRIC_LABELS = {
    "AUROC": "AUROC",
    "AUPRC": "AUPRC",
    "Accuracy": "Accuracy",
    "Balanced_Accuracy": "Balanced Accuracy",
    "F1": "F1",
    "F1_Macro": "F1 Macro",
    "Precision": "Precision",
    "Recall": "Recall",
    "Specificity": "Specificity",
    "FPR": "FPR",
    "ECE": "ECE",
    "BA@FPR0.10": r"BA@FPR$=0.10$",
}


def _is_nan(x):
    return x is None or (isinstance(x, float) and x != x)


def latex_metric(value, ci):
    if _is_nan(value):
        return "--"
    v = float(value)
    if ci is None:
        return f"{v:.3f}"
    lo, hi = float(ci[0]), float(ci[1])
    if _is_nan(lo) or _is_nan(hi):
        return f"{v:.3f}"
    return f"{v:.3f} ({lo:.3f}--{hi:.3f})"


def mode_label(mode):
    return {"baseline": "Baseline",
            "privileged_distillation": "Privileged"}.get(mode, str(mode))


def _lookup(metrics_dict, cis_dict, key, metric):
    if key not in metrics_dict:
        return None, None
    sub = metrics_dict[key]
    if metric not in sub:
        return None, None
    ci = cis_dict.get(key, {}).get(metric)
    return sub[metric], ci


def make_latex_table_transposed(
    metrics_odir, cis_odir,
    metrics_strict, cis_strict,
    metrics_camera, cis_camera,
    MODES, sides,
    conf,
    output_file="results_comparison.tex",
    
):
    lines = []
    combinations = [(m, s) for s in sides for m in MODES]
    n_cols = 1 + len(combinations)

    lines += [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{l" + "c" * len(combinations) + r"}",
        r"\toprule",
    ]

    # Header
    header = [r"\textbf{Metric}"]
    for mode, side in combinations:
        header.append(rf"\textbf{{{mode_label(mode)}}} \textbf{{({side.capitalize()})}}")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # ---- Internal ODIR test ----
    lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textbf{{Internal ODIR test}}}} \\")
    lines.append(r"\midrule")
    for metric in METRICS:
        row = [rf"\textbf{{{METRIC_LABELS[metric]}}}"]
        for mode, side in combinations:
            v, ci = _lookup(metrics_odir, cis_odir, (side, mode), metric)
            row.append(latex_metric(v, ci))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # ---- External strict ----
    lines.append(
        rf"\multicolumn{{{n_cols}}}{{l}}"
        r"{\textbf{External BRSET -- Strict (threshold from ODIR)}} \\"
    )
    lines.append(r"\midrule")
    for metric in METRICS:
        row = [rf"\textbf{{{METRIC_LABELS[metric]}}}"]
        for mode, side in combinations:
            vals = []
            for d in (1, 2):
                v, ci = _lookup(metrics, cis_strict, (side, mode, 'brset', d), metric)
                vals.append(latex_metric(v, ci))
            row.append(" / ".join(vals))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # ---- External camera-calibrated ----
    lines.append(
        rf"\multicolumn{{{n_cols}}}{{l}}"
        r"{\textbf{External BRSET -- Camera-calibrated "
        r"(threshold from Canon $d{=}0$)}} \\"
    )
    lines.append(r"\midrule")
    for metric in METRICS:
        row = [rf"\textbf{{{METRIC_LABELS[metric]}}}"]
        for mode, side in combinations:
            vals = []
            for d in (1, 2):
                v, ci = _lookup(metrics, cis, (side, mode, 'brset', d), metric)
                vals.append(latex_metric(v, ci))
            row.append(" / ".join(vals))
        lines.append(" & ".join(row) + r" \\")

    try:
        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{", f"{conf.task}: Performance of the baseline and privileged-distillation "
            r"students. Each cell in the external blocks shows "
            r"\emph{split 1 / split 2}: Nikon abnormal image field and Nikon "
            r"normal image field. \emph{Strict}: decision threshold fixed from "
            r"the ODIR test set. \emph{Camera-calibrated}: threshold re-selected "
            r"on the Canon split ($d{=}0$) and applied to the Nikon splits. "
            r"Values are point estimates with 95\% bootstrap confidence intervals.}",
            r"\label{tab:model_eye_comparison_transposed}",
            r"\end{table*}",
        ]

    except:
        pass
        
    latex = "\n".join(lines)
    Path(output_file).write_text(latex, encoding="utf-8")
    print(f"LaTeX table saved to: {output_file}")
    return latex


# ============================================================
# RUN
# ============================================================
latex = make_latex_table_transposed(
    metrics_odir=metrics,
    cis_odir=cis,
    metrics_strict= metrics,
    cis_strict= cis,
    metrics_camera= None,
    cis_camera=None,
    MODES=MODES,
    sides=sides,
    conf=conf,
    output_file=f"{conf.pref()}_results_comparison.tex",
)

print(latex)