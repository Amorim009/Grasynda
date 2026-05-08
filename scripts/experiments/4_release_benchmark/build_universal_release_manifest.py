"""
Build one canonical release manifest for the paper pipeline.

This does not generate data. It joins already-materialized release manifests and
skips rows whose release CSVs are missing, writing those skipped rows to a
separate report.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
RELEASE_ROOT = os.path.join(PROJECT_ROOT, "assets", "results", "release_benchmark")

DA_MANIFEST_PATH = os.path.join(RELEASE_ROOT, "release_all_20260505", "release_manifest.csv")
GRASYNDA_MANIFEST_PATH = os.path.join(RELEASE_ROOT, "release_all_grasynda_optimized_20260506", "release_manifest.csv")
FAITHFUL_MANIFEST_PATH = os.path.join(
    RELEASE_ROOT,
    "release_faithful_lpa_fpa_eps_sweep_k30_orthofpa_20260508",
    "release_manifest.csv",
)
OUTPUT_DIR = os.path.join(RELEASE_ROOT, "release_universal_grasynda_da_faithful_lpa_fpa_20260508")

DA_METHOD_ORDER = ["SeasonalMBB", "Jittering", "Scaling", "MagnitudeWarping", "TimeWarping", "DBA", "TSMixup", "TimeVAE"]
FAITHFUL_METHOD_ORDER = ["LPA", "FPA"]
SORT_GROUPS = ["Baseline", "Grasynda", "OtherAugmentation", "AnonymizedOriginal"]
REQUIRED_PATH_COLUMNS = ["Real_Train_Path", "Real_Test_Path", "Released_Train_Path"]


def read_manifest(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} manifest not found: {path}")
    df = pd.read_csv(path)
    df["Release_Source"] = label
    df["Release_Source_Manifest"] = os.path.abspath(path)
    return df


def select_rows(da_df: pd.DataFrame, grasynda_df: pd.DataFrame, faithful_df: pd.DataFrame) -> pd.DataFrame:
    baseline = da_df[(da_df["Family"] == "Baseline") & (da_df["Method"].isin(["RealTrain", "OriginalBaseline"]))].copy()
    baseline["Method"] = "OriginalBaseline"
    baseline["Variant_Name"] = "OriginalBaseline"
    grasynda = grasynda_df[(grasynda_df["Family"] == "Grasynda") & (grasynda_df["Method"] == "Grasynda_Optimized")].copy()
    augmentation = da_df[da_df["Family"] == "OtherAugmentation"].copy()
    faithful = faithful_df[(faithful_df["Family"] == "AnonymizedOriginal") & faithful_df["Method"].isin(FAITHFUL_METHOD_ORDER)].copy()
    return pd.concat([baseline, grasynda, augmentation, faithful], ignore_index=True)


def split_existing_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exists = pd.Series(True, index=df.index)
    missing_detail = []
    for col in REQUIRED_PATH_COLUMNS:
        col_exists = df[col].astype(str).map(os.path.exists)
        for idx in df.index[~col_exists]:
            missing_detail.append({"Row_Index": int(idx), "Missing_Column": col, "Missing_Path": df.at[idx, col]})
        exists &= col_exists
    kept = df[exists].copy()
    skipped = df[~exists].copy()
    if missing_detail:
        detail = pd.DataFrame(missing_detail)
        skipped = skipped.merge(detail, left_index=True, right_on="Row_Index", how="left")
    return kept, skipped


def method_sort_rank(row: pd.Series) -> int:
    if row["Family"] == "OtherAugmentation":
        return DA_METHOD_ORDER.index(row["Method"]) if row["Method"] in DA_METHOD_ORDER else len(DA_METHOD_ORDER)
    if row["Family"] == "AnonymizedOriginal":
        return FAITHFUL_METHOD_ORDER.index(row["Method"]) if row["Method"] in FAITHFUL_METHOD_ORDER else len(FAITHFUL_METHOD_ORDER)
    return 0


def sorted_manifest(df: pd.DataFrame) -> pd.DataFrame:
    dataset_order = {
        item: rank
        for rank, item in enumerate(df[["Dataset", "Group"]].drop_duplicates().itertuples(index=False, name=None))
    }
    out = df.copy()
    out["_Dataset_Rank"] = [dataset_order[(row.Dataset, row.Group)] for row in out.itertuples()]
    out["_Family_Rank"] = out["Family"].apply(lambda value: SORT_GROUPS.index(value) if value in SORT_GROUPS else len(SORT_GROUPS))
    out["_Method_Rank"] = out.apply(method_sort_rank, axis=1)
    out["_Epsilon_Rank"] = out["Epsilon"].fillna(-1.0).astype(float)
    out = out.sort_values(["_Dataset_Rank", "_Family_Rank", "_Method_Rank", "_Epsilon_Rank"]).reset_index(drop=True)
    return out.drop(columns=["_Dataset_Rank", "_Family_Rank", "_Method_Rank", "_Epsilon_Rank"])


def main() -> None:
    da_df = read_manifest(DA_MANIFEST_PATH, "release_all_20260505_da")
    grasynda_df = read_manifest(GRASYNDA_MANIFEST_PATH, "release_all_grasynda_optimized_20260506")
    faithful_df = read_manifest(FAITHFUL_MANIFEST_PATH, "release_faithful_lpa_fpa_orthofpa")
    selected = select_rows(da_df, grasynda_df, faithful_df)
    manifest, skipped = split_existing_rows(selected)
    manifest = sorted_manifest(manifest)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUTPUT_DIR, "release_manifest.csv")
    counts_path = os.path.join(OUTPUT_DIR, "release_manifest_counts.csv")
    skipped_path = os.path.join(OUTPUT_DIR, "release_manifest_skipped_missing_files.csv")
    config_path = os.path.join(OUTPUT_DIR, "release_manifest_config.json")

    manifest.to_csv(manifest_path, index=False)
    counts = manifest.groupby(["Family", "Method"], dropna=False).size().reset_index(name="Rows").sort_values(["Family", "Method"])
    counts.to_csv(counts_path, index=False)
    skipped.to_csv(skipped_path, index=False)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "da_manifest_path": os.path.abspath(DA_MANIFEST_PATH),
                "grasynda_manifest_path": os.path.abspath(GRASYNDA_MANIFEST_PATH),
                "faithful_manifest_path": os.path.abspath(FAITHFUL_MANIFEST_PATH),
                "output_manifest_path": os.path.abspath(manifest_path),
                "row_count": int(len(manifest)),
                "skipped_missing_file_rows": int(len(skipped)),
            },
            handle,
            indent=2,
        )

    print("### DONE ###", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Counts:   {counts_path}", flush=True)
    print(f"Skipped:  {skipped_path}", flush=True)
    print(counts.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
