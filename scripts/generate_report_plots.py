"""
generate_report_plots.py

Generates Kaplan-Meier survival curves and summary statistics
from the per_patient_*.csv produced by the orchestrator evaluation.

No modifications to the orchestrator code are needed.

Outputs:
  <output-dir>/km_orchestrator_<cohort>.pdf       KM stratified by risk class
  <output-dir>/km_complete_vs_missing_<cohort>.pdf KM complete vs missing patients
  <output-dir>/reliability_scatter_<cohort>.pdf    risk score vs reliability indicators
  <output-dir>/../summary_table.csv               machine-readable summary
  <output-dir>/../summary_table.tex               LaTeX-ready table

Usage:
    python3 scripts/generate_report_plots.py \\
        --run-dir results/evaluation/runs/t0_849885_20260519_191728 \\
        --output-dir figures

Dependencies: lifelines matplotlib pandas numpy
Install: pip install lifelines matplotlib pandas numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

RISK_COLORS = {"low": "#2ca02c", "medium": "#ff7f0e", "high": "#d62728"}


def km_plot_by_risk_class(df: pd.DataFrame, title: str, output_path: Path) -> dict:
    fig, ax = plt.subplots(figsize=(6, 5))

    groups = [c for c in ["low", "medium", "high"] if (df["risk_class"] == c).any()]
    for cls in groups:
        mask = df["risk_class"] == cls
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=f"{cls} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, color=RISK_COLORS[cls], ci_show=True)

    if len(groups) >= 2:
        test = multivariate_logrank_test(df["time"], df["risk_class"], df["event"])
        p_val = float(test.p_value)
        ax.text(0.05, 0.05, f"Log-rank p = {p_val:.3g}", transform=ax.transAxes, fontsize=10)
    else:
        p_val = float("nan")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    return {
        "logrank_p": p_val,
        **{f"n_{c}": int((df["risk_class"] == c).sum()) for c in ["low", "medium", "high"]},
    }


def km_plot_complete_vs_missing(df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    complete_mask = df["n_missing"] == 0

    for label, mask, color in [
        (f"Complete (n={complete_mask.sum()})", complete_mask, "#1f77b4"),
        (f"Missing modality (n={(~complete_mask).sum()})", ~complete_mask, "#d62728"),
    ]:
        if mask.sum() == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, "time"], df.loc[mask, "event"], label=label)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def reliability_scatter(df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    specs = [
        ("provenance", "Provenance (real features / total)", "risk score vs provenance"),
        ("mahal_pct", "Mahalanobis OOD percentile", "risk score vs OOD percentile"),
        ("ci_width", "Bootstrap 95% CI width", "risk score vs CI width"),
    ]
    for ax, (col, xlabel, subtitle) in zip(axes, specs):
        ax.scatter(df[col], df["risk_score"], alpha=0.5, s=20, c="#1f77b4")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Risk score")
        ax.set_title(subtitle)
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate report figures from per_patient CSV files")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run directory containing per_patient_*.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"),
                        help="Output directory for figures (default: figures/)")
    parser.add_argument("--cohorts", nargs="+", default=["luad", "lusc"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for cohort in args.cohorts:
        csv_path = args.run_dir / f"per_patient_{cohort}.csv"
        if not csv_path.exists():
            print(f"[WARN] {csv_path} not found, skipping {cohort}")
            continue

        df = pd.read_csv(csv_path)
        print(f"\n=== {cohort.upper()} ({len(df)} patients) ===")
        print(f"  Risk classes : {df['risk_class'].value_counts().to_dict()}")
        print(f"  Events       : {int(df['event'].sum())}/{len(df)}")
        print(f"  Median follow-up: {df['time'].median():.0f} days")
        print(f"  Missing-modality patients: {int((df['n_missing'] > 0).sum())}")

        km_path = args.output_dir / f"km_orchestrator_{cohort}.pdf"
        km_info = km_plot_by_risk_class(
            df,
            f"{cohort.upper()} — orchestrator risk stratification (T=0, N=3)",
            km_path,
        )
        print(f"  -> {km_path}  (log-rank p={km_info['logrank_p']:.3g})")

        cm_path = args.output_dir / f"km_complete_vs_missing_{cohort}.pdf"
        km_plot_complete_vs_missing(
            df,
            f"{cohort.upper()} — complete vs missing-modality patients",
            cm_path,
        )
        print(f"  -> {cm_path}")

        scatter_path = args.output_dir / f"reliability_scatter_{cohort}.pdf"
        reliability_scatter(
            df,
            f"{cohort.upper()} — risk score vs reliability indicators",
            scatter_path,
        )
        print(f"  -> {scatter_path}")

        summary_rows.append({
            "cohort": cohort,
            "n_patients": len(df),
            "n_complete": int((df["n_missing"] == 0).sum()),
            "n_missing_modality": int((df["n_missing"] > 0).sum()),
            "n_events": int(df["event"].sum()),
            "median_followup_days": float(df["time"].median()),
            "logrank_p": km_info["logrank_p"],
            "mean_provenance": float(df["provenance"].mean()),
            "mean_mahal_pct": float(df["mahal_pct"].mean()),
            "mean_ci_width": float(df["ci_width"].mean()),
        })

    if not summary_rows:
        print("\n[ERROR] No CSV files found. Check --run-dir.")
        return 1

    summary_df = pd.DataFrame(summary_rows)

    csv_out = args.output_dir / "summary_table.csv"
    summary_df.to_csv(csv_out, index=False)
    print(f"\nSummary CSV  -> {csv_out}")

    tex_out = args.output_dir / "summary_table.tex"
    cols = list(summary_df.columns)
    col_fmt = "l" + "r" * (len(cols) - 1)
    lines = [
        r"\begin{tabular}{" + col_fmt + r"}",
        r"\toprule",
        " & ".join(str(c) for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, row in summary_df.iterrows():
        def _fmt(v):
            try:
                return f"{float(v):.3f}"
            except (ValueError, TypeError):
                return str(v)
        lines.append(" & ".join(_fmt(v) for v in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_out.write_text("\n".join(lines) + "\n")
    print(f"Summary LaTeX -> {tex_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
