"""
Run NHITS TSTR forecasting on pre-materialized released train sets.

Configuration lives in the block below. Released-data rows use TSTR: train on
released data and test on the real held-out split. OriginalBaseline is the TRTR
anchor for sanity checks.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
from functools import partial
from neuralforecast import NeuralForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mase


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import run_universal_experiments_gpu as universal
from utils.config_gpu import (
    ACCELERATOR,
    REQUIRE_GPU,
    TORCH_DEVICE,
    assert_runtime_ready_for_requested_accelerator,
    get_runtime_hardware_summary,
)


# =========================
# Configuration
# =========================

MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "results",
    "release_benchmark",
    "release_universal_grasynda_da_faithful_lpa_fpa_20260508",
    "release_manifest.csv",
)
FORECASTING_MODELS = [
    "NHITS",
]
MAX_STEPS = 1000
EPSILON_FILTER = [0.48, 24.0]
FAMILIES_TO_RUN = [
    "Baseline",
    "AnonymizedOriginal",
]
METHODS_TO_RUN = [
    "OriginalBaseline",
    "FPA",
]
OUTPUT_TAG = "forecasting_baseline_faithful_fpa_eps048_24_20260508"


def normalize_ds_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ds" not in out.columns:
        return out
    ds = out["ds"]
    numeric = pd.to_numeric(ds, errors="coerce")
    if numeric.notna().all():
        if np.allclose(numeric.to_numpy(dtype=float), np.round(numeric.to_numpy(dtype=float))):
            out["ds"] = numeric.astype(int)
        else:
            out["ds"] = numeric.astype(float)
        return out
    parsed = pd.to_datetime(ds, errors="coerce")
    if parsed.notna().all():
        out["ds"] = parsed
    return out


def load_frame(path: str) -> pd.DataFrame:
    return normalize_ds_column(pd.read_csv(path))


def compute_mase_for_release(row: pd.Series, model_name: str, max_steps: int) -> float:
    real_train = load_frame(row["Real_Train_Path"])
    real_test = load_frame(row["Real_Test_Path"])
    released_train = load_frame(row["Released_Train_Path"])
    horizon = int(row["Horizon"])
    n_lags = int(row["NLags"])
    freq_str = row["Freq_Str"]
    freq_int = int(row["Freq_Int"])

    model_inst = universal.build_model_instance(
        model_name=model_name,
        horizon=horizon,
        n_lags=n_lags,
        max_steps=max_steps,
        alias=f"{row['Variant_Name']}_{model_name}",
    )
    nf = NeuralForecast(models=[model_inst], freq=freq_str)

    if row["Family"] == "Baseline":
        nf.fit(df=real_train, val_size=horizon)
        fcst = nf.predict()
    else:
        nf.fit(df=released_train, val_size=horizon)
        fcst = nf.predict(df=real_train)

    fcst_out = fcst.reset_index()
    pred_cols = [col for col in fcst_out.columns if col not in {"unique_id", "ds"}]
    if len(pred_cols) != 1:
        raise ValueError(f"Expected exactly 1 prediction column, got: {pred_cols}")
    pred_col = pred_cols[0]

    test_with_fcst = real_test.merge(fcst_out, on=["unique_id", "ds"], how="left")
    eval_df = evaluate(test_with_fcst, [partial(mase, seasonality=freq_int)], train_df=real_train)
    return float(eval_df.query('metric=="mase"')[pred_col].mean())


def main() -> None:
    runtime = get_runtime_hardware_summary()
    print("Runtime:")
    print(f"  Requested accelerator: {runtime['requested_accelerator']}")
    print(f"  NeuralForecast accelerator: {ACCELERATOR}")
    print(f"  Torch device: {TORCH_DEVICE}")
    print(f"  Torch CUDA available: {runtime['torch_cuda_available']}")
    print(f"  Torch CUDA device count: {runtime['torch_cuda_device_count']}")
    print(f"  GPU required: {REQUIRE_GPU}")
    assert_runtime_ready_for_requested_accelerator()

    manifest_path = os.path.abspath(MANIFEST_PATH)
    manifest = pd.read_csv(manifest_path)
    if FAMILIES_TO_RUN:
        manifest = manifest[manifest["Family"].isin(FAMILIES_TO_RUN)].copy()
    if METHODS_TO_RUN:
        manifest = manifest[manifest["Method"].isin(METHODS_TO_RUN)].copy()
    if EPSILON_FILTER is not None:
        epsilons = {float(value) for value in EPSILON_FILTER}
        non_anonymized_mask = manifest["Family"] != "AnonymizedOriginal"
        epsilon_mask = manifest["Epsilon"].astype(float).isin(epsilons)
        manifest = manifest[non_anonymized_mask | epsilon_mask].copy()
        if manifest.empty:
            raise ValueError(f"No rows remain after epsilon filter: {sorted(epsilons)}")
    expected_rows = len(manifest)
    print(f"Rows selected from manifest: {expected_rows}", flush=True)

    base_dir = os.path.dirname(os.path.dirname(manifest_path))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = OUTPUT_TAG or f"forecasting_{timestamp}"
    out_dir = os.path.join(base_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "forecasting_config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": timestamp,
                "manifest": manifest_path,
                "forecasting_models": FORECASTING_MODELS,
                "max_steps": MAX_STEPS,
                "epsilon_filter": EPSILON_FILTER,
                "families_to_run": FAMILIES_TO_RUN,
                "methods_to_run": METHODS_TO_RUN,
                "forecasting_protocol": "TSTR for released rows; TRTR only for OriginalBaseline",
            },
            handle,
            indent=2,
        )

    rows = []
    for _, release_row in manifest.iterrows():
        print(f"\n### {release_row['Dataset']} - {release_row['Group']} | {release_row['Variant_Name']} ###", flush=True)
        for model_name in FORECASTING_MODELS:
            start = time.time()
            try:
                mase_score = compute_mase_for_release(release_row, model_name, MAX_STEPS)
                rows.append(
                    {
                        "Dataset": release_row["Dataset"],
                        "Group": release_row["Group"],
                        "Family": release_row["Family"],
                        "Method": release_row["Method"],
                        "Variant_Name": release_row["Variant_Name"],
                        "Epsilon": release_row["Epsilon"],
                        "Seed": release_row["Seed"],
                        "Forecasting_Model": model_name,
                        "Evaluation_Protocol": (
                            "TRTR_OriginalBaseline_TestOnReal"
                            if release_row["Family"] == "Baseline"
                            else "TSTR_TrainOnReleased_TestOnReal"
                        ),
                        "Utility_MASE": mase_score,
                        "Released_Size": release_row["Released_Size"],
                        "Time_Sec": time.time() - start,
                    }
                )
                print(f"  -> {model_name} Utility_MASE={mase_score:.4f}", flush=True)
            except Exception as exc:
                print(f"  -> {model_name} failed: {exc}", flush=True)
        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(out_dir, "forecasting_results_checkpoint.csv"), index=False)

    if not rows:
        raise RuntimeError("No forecasting results were produced.")

    results_df = pd.DataFrame(rows)
    results_path = os.path.join(out_dir, "forecasting_results_detailed.csv")
    results_df.to_csv(results_path, index=False)

    summary_df = (
        results_df.groupby(
            ["Family", "Method", "Variant_Name", "Epsilon", "Forecasting_Model", "Evaluation_Protocol"],
            dropna=False,
        )[["Utility_MASE", "Time_Sec"]]
        .mean()
        .reset_index()
        .sort_values(["Forecasting_Model", "Evaluation_Protocol", "Utility_MASE"])
    )
    summary_path = os.path.join(out_dir, "forecasting_results_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n### DONE ###", flush=True)
    print(f"Detailed: {results_path}", flush=True)
    print(f"Summary:  {summary_path}", flush=True)


if __name__ == "__main__":
    main()
