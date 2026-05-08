import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INPUT_DIR = PROJECT_ROOT / "assets" / "results" / "grasynda_short_series_per_series"
OUTPUT_DIR = INPUT_DIR / "plots"
METHOD_ORDER = ["Baseline", "NoEnsemble"]
METHOD_LABELS = {
    "Baseline": "Baseline (No Aug)",
    "NoEnsemble": "Grasynda NoEnsemble",
}
METHOD_LINESTYLES = {
    "Baseline": "-",
    "NoEnsemble": "--",
}
METHOD_MARKERS = {
    "Baseline": "o",
    "NoEnsemble": "s",
}
Y_CLIP_QUANTILE = 0.975
DELTA_CLIP_QUANTILE = 0.975
N_BINS = 12
ANOMALY_IQR_MULTIPLIER = 3.0
DELTA_WINDOW_LOWER_Q = 0.05
DELTA_WINDOW_UPPER_Q = 0.95
DATASET_COLORS = {
    "Gluonts / m1_monthly": "#1f4e79",
    "Gluonts / m1_quarterly": "#d17a22",
    "M3 / Monthly": "#207561",
    "M3 / Quarterly": "#b33f62",
    "NN3 / Monthly": "#5b4b8a",
    "Tourism / Monthly": "#0f8b8d",
    "Tourism / Quarterly": "#7a5c3e",
}


def find_latest_per_series_csv() -> Path:
    patterns = [
        "grasynda_baseline_vs_noensemble_per_series_mase_*.csv",
        "grasynda_noensemble_per_series_mase_*.csv",
        "grasynda_*_per_series_mase_*.csv",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(INPUT_DIR.glob(pattern))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No compatible per-series CSV found in {INPUT_DIR}")
    return candidates[0]


def build_long_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dataset_group"] = df["Dataset"] + " / " + df["Group"]

    frames = []
    if "MASE_Baseline" in df.columns:
        base = df[
            [
                "unique_id",
                "Dataset",
                "Group",
                "dataset_group",
                "total_length",
                "train_length",
                "short_threshold",
                "is_short",
                "series_bucket",
                "MASE_Baseline",
            ]
        ].copy()
        base["Method"] = "Baseline"
        base["MASE"] = base.pop("MASE_Baseline")
        frames.append(base)

    if "MASE_NoEnsemble" in df.columns:
        noens = df[
            [
                "unique_id",
                "Dataset",
                "Group",
                "dataset_group",
                "total_length",
                "train_length",
                "short_threshold",
                "is_short",
                "series_bucket",
                "MASE_NoEnsemble",
            ]
        ].copy()
        noens["Method"] = "NoEnsemble"
        noens["MASE"] = noens.pop("MASE_NoEnsemble")
        frames.append(noens)

    if "MASE" in df.columns:
        single = df[
            [
                "unique_id",
                "Dataset",
                "Group",
                "dataset_group",
                "total_length",
                "train_length",
                "short_threshold",
                "is_short",
                "series_bucket",
                "MASE",
            ]
        ].copy()
        single["Method"] = "NoEnsemble"
        frames.append(single)

    if not frames:
        raise ValueError("Could not find compatible MASE columns in the input CSV.")

    long_df = pd.concat(frames, ignore_index=True)
    long_df["Method"] = pd.Categorical(long_df["Method"], categories=METHOD_ORDER, ordered=True)
    return long_df.sort_values(["dataset_group", "Method", "total_length", "unique_id"]).reset_index(drop=True)


def build_delta_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not {"MASE_Baseline", "MASE_NoEnsemble"}.issubset(df.columns):
        raise ValueError("Delta plot requires both MASE_Baseline and MASE_NoEnsemble columns.")

    delta_df = df.copy()
    delta_df["dataset_group"] = delta_df["Dataset"] + " / " + delta_df["Group"]
    delta_df["delta"] = delta_df["MASE_NoEnsemble"] - delta_df["MASE_Baseline"]
    return delta_df[
        [
            "unique_id",
            "Dataset",
            "Group",
            "dataset_group",
            "total_length",
            "train_length",
            "short_threshold",
            "is_short",
            "series_bucket",
            "delta",
        ]
    ].copy()


def clip_mase_for_plotting(df_long: pd.DataFrame) -> pd.DataFrame:
    df_long = df_long.copy()
    clipped_parts = []
    for (dataset_group, method), sub in df_long.groupby(["dataset_group", "Method"], observed=True):
        upper = sub["MASE"].quantile(Y_CLIP_QUANTILE)
        clipped = sub.copy()
        clipped["MASE_plot"] = clipped["MASE"].clip(upper=upper)
        clipped["clip_upper"] = upper
        clipped_parts.append(clipped)
    return pd.concat(clipped_parts, ignore_index=True)


def clip_delta_for_plotting(delta_df: pd.DataFrame) -> pd.DataFrame:
    delta_df = delta_df.copy()
    clipped_parts = []
    for dataset_group, sub in delta_df.groupby("dataset_group"):
        upper = sub["delta"].abs().quantile(DELTA_CLIP_QUANTILE)
        clipped = sub.copy()
        clipped["delta_plot"] = clipped["delta"].clip(lower=-upper, upper=upper)
        clipped["clip_abs_upper"] = upper
        clipped_parts.append(clipped)
    return pd.concat(clipped_parts, ignore_index=True)


def annotate_delta_anomalies(delta_df: pd.DataFrame) -> pd.DataFrame:
    delta_df = delta_df.copy()
    annotated_parts = []
    for dataset_group, sub in delta_df.groupby("dataset_group"):
        q1 = sub["delta"].quantile(0.25)
        q3 = sub["delta"].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - ANOMALY_IQR_MULTIPLIER * iqr
        upper = q3 + ANOMALY_IQR_MULTIPLIER * iqr

        annotated = sub.copy()
        annotated["delta_anomaly_lower"] = lower
        annotated["delta_anomaly_upper"] = upper
        annotated["is_delta_anomaly"] = (annotated["delta"] < lower) | (annotated["delta"] > upper)
        annotated_parts.append(annotated)
    return pd.concat(annotated_parts, ignore_index=True)


def build_binned_curve(df: pd.DataFrame, value_col: str, n_bins: int = N_BINS) -> pd.DataFrame:
    curve = df[["total_length", value_col]].dropna().copy()
    if len(curve) < 3 or curve["total_length"].nunique() < 3:
        return pd.DataFrame()

    curve["bin"] = pd.qcut(
        curve["total_length"].rank(method="first"),
        q=min(n_bins, len(curve)),
        duplicates="drop",
    )

    grouped = (
        curve.groupby("bin", observed=True)
        .agg(
            total_length=("total_length", "median"),
            value_median=(value_col, "median"),
            n_points=(value_col, "size"),
        )
        .reset_index(drop=True)
    )
    return grouped[grouped["n_points"] > 0]


def fit_trendline(x: pd.Series, y: pd.Series):
    if x.nunique() < 2:
        return None, None
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    return (x_line, y_line), slope


def save_figure_safe(fig: plt.Figure, output_path: Path, dpi: int | None = None) -> Path:
    try:
        if dpi is None:
            fig.savefig(output_path)
        else:
            fig.savefig(output_path, dpi=dpi)
        return output_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
        if dpi is None:
            fig.savefig(fallback)
        else:
            fig.savefig(fallback, dpi=dpi)
        print(f"Warning: could not overwrite {output_path.name}; saved {fallback.name} instead.")
        return fallback


def add_shared_legend(ax: plt.Axes, include_methods: bool = True) -> None:
    dataset_handles = []
    for dataset_group in sorted(DATASET_COLORS.keys()):
        dataset_handles.append(
            mlines.Line2D([], [], color=DATASET_COLORS[dataset_group], lw=2.6, label=dataset_group)
        )

    dataset_legend = ax.legend(
        handles=dataset_handles,
        title="Dataset",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(dataset_legend)

    if include_methods:
        method_handles = [
            mlines.Line2D(
                [],
                [],
                color="#333333",
                lw=0,
                marker=METHOD_MARKERS[method],
                ms=6.0,
                label=METHOD_LABELS[method],
            )
            for method in METHOD_ORDER
        ]
        ax.legend(
            handles=method_handles,
            title="Method",
            fontsize=8,
            title_fontsize=9,
            frameon=True,
            loc="upper left",
            bbox_to_anchor=(1.01, 0.54),
        )


def plot_actual_mase(df_long: pd.DataFrame, output_png: Path, output_pdf: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    dataset_groups = df_long["dataset_group"].drop_duplicates().tolist()

    for dataset_group in dataset_groups:
        sub = df_long[df_long["dataset_group"] == dataset_group]
        color = DATASET_COLORS.get(dataset_group, "#444444")

        for method in [m for m in METHOD_ORDER if m in sub["Method"].astype(str).unique()]:
            method_sub = sub[sub["Method"] == method]
            curve = build_binned_curve(method_sub, "MASE_plot")
            if curve.empty:
                continue

            ax.scatter(
                curve["total_length"],
                curve["value_median"],
                color=color,
                s=52,
                marker=METHOD_MARKERS[method],
                alpha=0.95,
                edgecolors="white",
                linewidths=0.6,
            )

    ax.set_xscale("log")
    ax.set_title("Series Length vs Forecasting Error Across Datasets", fontsize=16)
    ax.set_xlabel("Total Time-Series Length", fontsize=12)
    ax.set_ylabel("Median Per-Series MASE", fontsize=12)
    ax.grid(True, alpha=0.22)
    add_shared_legend(ax, include_methods=True)
    ax.text(
        0.015,
        0.015,
        (
            f"Each point = median MASE within one of {N_BINS} within-dataset size bins.\n"
            f"x = median length of the bin. y clipped at p{int(Y_CLIP_QUANTILE * 100)} within each dataset-method for readability."
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    saved_png = save_figure_safe(fig, output_png, dpi=240)
    saved_pdf = save_figure_safe(fig, output_pdf)
    plt.close(fig)
    return saved_png, saved_pdf


def plot_delta(delta_df: pd.DataFrame, output_png: Path, output_pdf: Path) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    dataset_groups = delta_df["dataset_group"].drop_duplicates().tolist()

    for dataset_group in dataset_groups:
        sub = delta_df[delta_df["dataset_group"] == dataset_group]
        color = DATASET_COLORS.get(dataset_group, "#444444")
        curve = build_binned_curve(sub, "delta_plot")
        if curve.empty:
            continue

        ax.scatter(
            curve["total_length"],
            curve["value_median"],
            color=color,
            s=58,
            marker="o",
            alpha=0.98,
            edgecolors="white",
            linewidths=0.6,
            label=dataset_group,
        )

        line, _ = fit_trendline(sub["total_length"], sub["delta_plot"])
        if line[0] is not None:
            ax.plot(
                line[0],
                line[1],
                color=color,
                lw=2.6,
                ls=(0, (1.4, 2.0)),
                alpha=0.95,
            )

    ax.axhline(0.0, color="#444444", lw=1.2, ls="--", alpha=0.8)
    ax.set_xscale("log")
    ax.set_title("Relative Performance vs Series Length Across Datasets", fontsize=16)
    ax.set_xlabel("Total Time-Series Length", fontsize=12)
    ax.set_ylabel("Median Delta MASE (NoEnsemble - Baseline)", fontsize=12)
    ax.grid(True, alpha=0.22)

    dataset_handles = [
        mlines.Line2D([], [], color=DATASET_COLORS[dg], lw=2.6, marker="o", ms=4.5, label=dg)
        for dg in sorted(dataset_groups)
    ]
    trend_handles = [
        mlines.Line2D([], [], color="#333333", lw=0, marker="o", ms=6.5, label="Binned median delta"),
        mlines.Line2D([], [], color="#333333", lw=2.6, ls=(0, (1.4, 2.0)), label="Linear trend"),
        mlines.Line2D([], [], color="#444444", lw=1.2, ls="--", label="Zero line"),
    ]

    dataset_legend = ax.legend(
        handles=dataset_handles,
        title="Dataset",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(dataset_legend)
    ax.legend(
        handles=trend_handles,
        title="How To Read",
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.56),
    )

    ax.text(
        0.015,
        0.015,
        (
            f"Delta = NoEnsemble MASE - Baseline MASE.\n"
            f"Above 0: Baseline better. Below 0: Grasynda better.\n"
            f"Dots = median delta within each of {N_BINS} within-dataset size bins.\n"
            f"Trend lines = linear fit of clipped per-series deltas against total length.\n"
            f"Deltas clipped symmetrically at p{int(DELTA_CLIP_QUANTILE * 100)} of |delta| for readability."
        ),
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    saved_png = save_figure_safe(fig, output_png, dpi=240)
    saved_pdf = save_figure_safe(fig, output_pdf)
    plt.close(fig)
    return saved_png, saved_pdf


def save_pdfpages_safe(build_callback, output_path: Path) -> Path:
    try:
        with PdfPages(output_path) as pdf:
            build_callback(pdf)
        return output_path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
        with PdfPages(fallback) as pdf:
            build_callback(pdf)
        print(f"Warning: could not overwrite {output_path.name}; saved {fallback.name} instead.")
        return fallback


def plot_delta_per_dataset_pdf(delta_df: pd.DataFrame, output_pdf: Path) -> Path:
    dataset_groups = delta_df["dataset_group"].drop_duplicates().tolist()

    def build_pdf(pdf: PdfPages) -> None:
        for dataset_group in dataset_groups:
            sub = delta_df[delta_df["dataset_group"] == dataset_group].copy()
            color = DATASET_COLORS.get(dataset_group, "#444444")
            curve = build_binned_curve(sub, "delta")

            fig, ax = plt.subplots(figsize=(11.5, 7.0))

            ax.scatter(
                sub["total_length"],
                sub["delta"],
                color=color,
                s=18,
                alpha=0.16,
                edgecolors="none",
                label="Per-series delta",
            )

            if not curve.empty:
                ax.scatter(
                    curve["total_length"],
                    curve["value_median"],
                    color=color,
                    s=64,
                    alpha=0.98,
                    edgecolors="white",
                    linewidths=0.7,
                    label=f"Binned median ({N_BINS} bins)",
                    zorder=3,
                )

            line, slope = fit_trendline(sub["total_length"], sub["delta"])
            if line[0] is not None:
                ax.plot(
                    line[0],
                    line[1],
                    color="#000000",
                    lw=3.8,
                    ls="-",
                    alpha=0.98,
                    label=f"Linear trend (slope={slope:.4f})",
                    zorder=4,
                )

            threshold = sub["short_threshold"].dropna().iloc[0] if sub["short_threshold"].notna().any() else None
            if threshold is not None:
                ax.axvline(threshold, color="#666666", lw=1.2, ls=":", alpha=0.85)

            ax.axhline(0.0, color="#444444", lw=1.2, ls="--", alpha=0.85)
            ax.set_xscale("log")
            ax.set_title(f"{dataset_group}: Relative Performance vs Series Length", fontsize=15)
            ax.set_xlabel("Total Time-Series Length", fontsize=12)
            ax.set_ylabel("Delta MASE (NoEnsemble - Baseline)", fontsize=12)
            ax.grid(True, alpha=0.22)

            y_low = sub["delta"].quantile(DELTA_WINDOW_LOWER_Q)
            y_high = sub["delta"].quantile(DELTA_WINDOW_UPPER_Q)
            y_span = max(y_high - y_low, 0.18)
            y_pad = max(0.05, 0.14 * y_span)
            ax.set_ylim(y_low - y_pad, y_high + y_pad)

            ax.text(
                0.015,
                0.015,
                (
                    "Negative delta: Grasynda better. Positive delta: Baseline better.\n"
                    f"Dots: per-series deltas. Large filled circles: median delta in {N_BINS} size bins.\n"
                    "Trend: linear fit on all per-series deltas, including off-window extremes.\n"
                    f"Displayed y-window uses dataset quantiles q{int(DELTA_WINDOW_LOWER_Q*100)} to q{int(DELTA_WINDOW_UPPER_Q*100)} with padding.\n"
                    "Some extreme points may fall outside the visible window but still affect the line and medians."
                ),
                transform=ax.transAxes,
                fontsize=9,
                color="#555555",
                ha="left",
                va="bottom",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
            )

            ax.legend(fontsize=9, frameon=True, loc="upper left")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return save_pdfpages_safe(build_pdf, output_pdf)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Optional per-series CSV path.")
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else find_latest_per_series_csv()
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df_long = clip_mase_for_plotting(build_long_dataframe(df))
    delta_df = clip_delta_for_plotting(build_delta_dataframe(df))
    delta_df_for_pages = annotate_delta_anomalies(build_delta_dataframe(df))

    stem = input_path.stem.replace("_per_series_mase_", "_")
    actual_png = OUTPUT_DIR / f"size_vs_actual_mase_{stem}.png"
    actual_pdf = OUTPUT_DIR / f"size_vs_actual_mase_{stem}.pdf"
    delta_png = OUTPUT_DIR / f"size_vs_delta_mase_{stem}.png"
    delta_pdf = OUTPUT_DIR / f"size_vs_delta_mase_{stem}.pdf"
    delta_dataset_pdf = OUTPUT_DIR / f"size_vs_delta_mase_by_dataset_{stem}.pdf"

    saved_actual_png, saved_actual_pdf = plot_actual_mase(df_long, actual_png, actual_pdf)
    saved_delta_png, saved_delta_pdf = plot_delta(delta_df, delta_png, delta_pdf)
    saved_delta_dataset_pdf = plot_delta_per_dataset_pdf(delta_df_for_pages, delta_dataset_pdf)

    print(f"Input:  {input_path}")
    print(f"Saved:  {saved_actual_png}")
    print(f"Saved:  {saved_actual_pdf}")
    print(f"Saved:  {saved_delta_png}")
    print(f"Saved:  {saved_delta_pdf}")
    print(f"Saved:  {saved_delta_dataset_pdf}")


if __name__ == "__main__":
    main()
