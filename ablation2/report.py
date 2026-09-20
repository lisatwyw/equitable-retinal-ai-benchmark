# ============================================================
# REPORT.PY -- Dual-protocol transposed LaTeX table
# ============================================================
# Updated to match the dual-protocol training script:
#   metrics[side, mode]                : ODIR internal test
#   metrics_strict[side, mode, d]      : BRSET with ODIR-fixed threshold
#   metrics_camera[side, mode, d]      : BRSET with Canon-calibrated threshold
#
# External splits by convention:
#   d = 0 : Canon CR            (calibration only)
#   d = 1 : Nikon NF5050        (abnormal image field)
#   d = 2 : Nikon NF5050        (normal image field)
#
# Metrics are rows; model/eye combinations are columns.
# ============================================================

from pathlib import Path


METRICS = [
    "AUROC",
    "AUPRC",
    "Accuracy",
    "Balanced_Accuracy",
    "F1",
    "F1_Macro",
    "Precision",
    "Recall",
    "Specificity",
    "FPR",
    "ECE",
    "BA@FPR0.10",
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
    "BA@FPR0.10": "BA@FPR$=0.10$",
}


# ============================================================
# FORMATTERS
# ============================================================
def latex_metric(value, ci):
    """Format a metric with an optional 95% CI."""
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return "--"
    value = float(value)
    if ci is None:
        return f"{value:.3f}"
    lower = float(ci[0])
    upper = float(ci[1])
    if lower != lower or upper != upper:  # NaN
        return f"{value:.3f}"
    return f"{value:.3f} ({lower:.3f}--{upper:.3f})"


def mode_label(mode):
    if mode == "baseline":
        return "Baseline"
    if mode == "privileged_distillation":
        return "Privileged"
    return str(mode)


# ============================================================
# SAFE ACCESSOR
# ============================================================
def _safe_get(d, key, metric):
    """Return (value, ci) or (None, None) if the key or metric is missing."""
    if key not in d:
        return None, None
    sub = d[key]
    if metric not in sub:
        return None, None
    ci_key = key
    return sub[metric], None


def _get_ci(cis_dict, key, metric):
    if key not in cis_dict:
        return None
    sub = cis_dict[key]
    if metric not in sub:
        return None
    return sub[metric]


# ============================================================
# TABLE BUILDER
# ============================================================
def make_latex_table_transposed(
    metrics_odir,
    cis_odir,
    metrics_strict,
    cis_strict,
    metrics_camera,
    cis_camera,
    MODES,
    sides,
    DSET,
    output_file="results_comparison.tex",
):
    lines = []

    # Column order: for each side, baseline then privileged
    combinations = []
    for side in sides:
        for mode in MODES:
            combinations.append((mode, side))

    n_cols = 1 + len(combinations)

    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{l" + "c" * len(combinations) + r"}")
    lines.append(r"\toprule")

    # Header
    header = [r"\textbf{Metric}"]
    for mode, side in combinations:
        header.append(
            rf"\textbf{{{mode_label(mode)}}} \textbf{{({side.capitalize()})}}"
        )
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # ---- Internal test ----
    lines.append(
        rf"\multicolumn{{{n_cols}}}{{l}}"
        r"{\textbf{Internal ODIR test}} \\"
    )
    lines.append(r"\midrule")
    for metric in METRICS:
        row = [rf"\textbf{{{METRIC_LABELS[metric]}}}"]
        for mode, side in combinations:
            value, _ = _safe_get(metrics_odir, (side, mode), metric)
            ci = _get_ci(cis_odir, (side, mode), metric)
            row.append(latex_metric(value, ci))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # ---- External BRSET: strict threshold (from ODIR) ----
    lines.append(
        rf"\multicolumn{{{n_cols}}}{{l}}"
        r"{\textbf{External BRSET -- Strict (threshold from ODIR)}} \\"
    )
    lines.append(r"\midrule")
    for metric in METRICS:
        row = [rf"\textbf{{{METRIC_LABELS[metric]}}}"]
        for mode, side in combinations:
            # Report splits 1 and 2 (the Nikon splits) as the strict result.
            # Split 0 (Canon) was reserved for camera-calibration and is
            # not a held-out set under either protocol.
            vals = []
            for d in (1, 2):
                key = (side, mode, d)
                v, _ = _safe_get(metrics_strict, key, metric)
                ci = _get_ci(cis_strict, key, metric)
                vals.append(latex_metric(v, ci))
            row.append(" / ".join(vals))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

    # ---- External BRSET: camera-calibrated threshold ----
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
                key = (side, mode, d)
                v, _ = _safe_get(metrics_camera, key, metric)
                ci = _get_ci(cis_camera, key, metric)
                vals.append(latex_metric(v, ci))
            row.append(" / ".join(vals))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(
        r"\caption{Performance of the baseline and privileged-distillation "
        r"students. Internal ODIR test set is reported first. External BRSET "
        r"results are shown under two protocols. \emph{Strict}: the decision "
        r"threshold is fixed from the ODIR test set and applied unchanged to "
        r"the Nikon data ($d{=}1$ abnormal image field, $d{=}2$ normal image "
        r"field). \emph{Camera-calibrated}: the threshold is re-selected on "
        r"the Canon split ($d{=}0$) and applied to the same Nikon splits. "
        r"Each cell shows \emph{split 1 / split 2}. Values are point estimates "
        r"with 95\% bootstrap confidence intervals. Bold marks the best "
        r"result within each row block.}"
    )

    lines.append(r"\label{tab:model_eye_comparison_transposed}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    Path(output_file).write_text(latex, encoding="utf-8")
    print(f"LaTeX table saved to: {output_file}")
    return latex


# ============================================================
# RUN
# ============================================================
latex = make_latex_table_transposed(
    metrics_odir=metrics,        # metrics[(side, mode)]
    cis_odir=cis,                # cis[(side, mode)]
    metrics_strict=metrics_strict,   # metrics_strict[(side, mode, d)]
    cis_strict=cis_strict,
    metrics_camera=metrics_camera,   # metrics_camera[(side, mode, d)]
    cis_camera=cis_camera,
    MODES=MODES,
    sides=sides,
    DSET=DSET,
    output_file=f"{pref}_results_comparison.tex",
)

print(latex)