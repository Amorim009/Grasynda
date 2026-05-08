"""
Focused PyMDMA evaluation for the default Scaling generator.

This mirrors the default dataset-loading and synthetic-data workflow used in the
comprehensive privacy run, but limits execution to the Scaling method so we can
recompute those rows cleanly for plotting.
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.insert(0, project_root)

from src.workflow import ExpWorkflow
from utils.load_data.config import DATASETS

try:
    from pymdma.tabular.data.load import TabularDataset
    from pymdma.tabular.measures.synthesis_val import (
        Authenticity,
        DCRPrivacy,
        ImprovedPrecision,
        ImprovedRecall,
    )
except ImportError as e:
    print(f"PyMDMA import error: {e}")
    sys.exit(1)


METHOD_NAME = "Scaling"
N_SYNTH_PER_UID = 1
SEED = 42

DATASETS_TO_TEST = [
    ("M3", "Monthly"),
    ("M3", "Quarterly"),
    ("Gluonts", "m1_monthly"),
    ("Gluonts", "m1_quarterly"),
    ("NN3", "Monthly"),
    ("Tourism", "Monthly"),
    ("Tourism", "Quarterly"),
]

out_dir = os.path.join(
    project_root, "assets", "results", "pymdma_metrics", "corrected"
)
os.makedirs(out_dir, exist_ok=True)

FINAL_RESULTS_FILENAME = os.path.join(
    out_dir, "final_results_scaling_default_datasets.csv"
)
SUMMARY_FILENAME = os.path.join(
    out_dir, "summary_scaling_default_datasets.csv"
)


def load_series_as_lists(df: pd.DataFrame):
    series_list = []
    series_ids = []
    for uid, group_df in df.groupby("unique_id"):
        values = group_df.sort_values("ds")["y"].values
        series_list.append(values)
        series_ids.append(uid)
    return series_list, series_ids


def truncate_to_length(series_list, T: int) -> np.ndarray:
    return np.asarray([series[-T:] for series in series_list])


def lists_to_tabular(series_list, T: int, series_ids=None) -> pd.DataFrame:
    values = truncate_to_length(series_list, T=T)
    col_names = [f"t_{i}" for i in range(T)]
    return pd.DataFrame(values, columns=col_names, index=series_ids)


def extract_synthetic_only(augmented_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:
    real_ids = set(real_df["unique_id"].unique())
    synth_only = augmented_df[~augmented_df["unique_id"].isin(real_ids)].copy()
    if not synth_only.empty:
        overlap = set(synth_only["unique_id"].unique()) & real_ids
        if overlap:
            synth_only = synth_only[~synth_only["unique_id"].isin(overlap)]
    return synth_only


def evaluate_scaling():
    np.random.seed(SEED)

    auth = Authenticity()
    imp_prec = ImprovedPrecision()
    imp_rec = ImprovedRecall()
    dcr_priv = DCRPrivacy()

    all_results = []

    for ds_name, group in DATASETS_TO_TEST:
        print(f"\n### Evaluating {METHOD_NAME} on {ds_name} - {group} ###")
        data_loader = DATASETS[ds_name]
        df_real, horizon, _, freq_str, freq_int = data_loader.load_everything(group)
        real_list, real_ids = load_series_as_lists(df_real)

        max_len = df_real["unique_id"].value_counts().max() - (2 * horizon)
        min_len = df_real["unique_id"].value_counts().min() - (2 * horizon)
        n_uids = df_real["unique_id"].nunique()
        max_n_uids = max(2, int(np.round(np.log(n_uids), 0)))

        params = {
            "seas_period": freq_int,
            "freq": freq_str,
            "max_n_uids": max_n_uids,
            "min_len": min_len,
            "max_len": max_len,
        }

        start_time = time.time()
        augmented_df = ExpWorkflow.get_offline_augmented_data(
            df_real,
            METHOD_NAME,
            params,
            n_series_by_uid=N_SYNTH_PER_UID,
        )
        synth_df = extract_synthetic_only(augmented_df, df_real)
        if synth_df.empty:
            print("  [SKIP] Empty synthetic data")
            continue

        synth_list, synth_ids = load_series_as_lists(synth_df)
        lengths = [len(series) for series in real_list]
        lengths.extend(len(series) for series in synth_list)
        T = int(np.min(lengths))
        if T <= 1:
            raise ValueError(f"Common truncation length too small (T={T}) for {ds_name} {group}")

        real_eval = lists_to_tabular(real_list, T=T, series_ids=real_ids)
        synth_eval = lists_to_tabular(synth_list, T=T, series_ids=synth_ids)

        eval_loader = TabularDataset(
            file_path=None,
            data=real_eval,
            tag_name="real",
            scaler="standard",
            imputer="knn",
            with_onehot=False,
        )
        real_scaled = eval_loader.data_s
        _, _, synth_scaled, _ = eval_loader.transform(
            data=synth_eval, scale_fit=False, meta_fit=False
        )

        auth_s = auth.compute(real_scaled, synth_scaled).value[0]
        fid_s = imp_prec.compute(real_scaled, synth_scaled).value[0]
        div_s = imp_rec.compute(real_scaled, synth_scaled).value[0]
        priv_s = dcr_priv.compute(real_scaled, synth_scaled).value[0]["privacy"] / 100

        res = {
            "Dataset": ds_name,
            "Group": group,
            "Method": METHOD_NAME,
            "Authenticity": auth_s,
            "Fidelity": fid_s,
            "Diversity": div_s,
            "Privacy": priv_s,
            "Time_Sec": time.time() - start_time,
            "N_Synth": len(synth_eval),
            "Seed": SEED,
        }
        all_results.append(res)
        print(
            f"  [OK] Auth: {auth_s:.3f}, Fid: {fid_s:.3f}, Priv: {priv_s:.3f}, "
            f"Div: {div_s:.3f}, N_Synth: {len(synth_eval)}"
        )

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(FINAL_RESULTS_FILENAME, index=False)

    summary_df = (
        final_df.groupby("Method")[["Authenticity", "Fidelity", "Diversity", "Privacy"]]
        .mean()
        .reset_index()
    )
    summary_df["Seed"] = SEED
    summary_df.to_csv(SUMMARY_FILENAME, index=False)

    print(f"\nSaved results to {FINAL_RESULTS_FILENAME}")
    print(f"Saved summary to {SUMMARY_FILENAME}")


if __name__ == "__main__":
    evaluate_scaling()
