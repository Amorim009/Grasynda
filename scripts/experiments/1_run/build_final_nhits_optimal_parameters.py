"""
Build a single audited NHITS optimal-parameter file from random-search outputs.

The source of truth is the latest per-target *_best_config.json file for each
augmentation method, not the latest consolidated snapshot, because consolidated
files may be partial progress saves.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEARCH_ROOT = REPO_ROOT / "assets" / "results" / "random_search"
OUTPUT_PATH = RANDOM_SEARCH_ROOT / "nhits_optimal_parameters_final.json"

TARGETS = [
    ("Gluonts", "m1_monthly"),
    ("Gluonts", "m1_quarterly"),
    ("M3", "Monthly"),
    ("M3", "Quarterly"),
    ("NN3", "Monthly"),
    ("Tourism", "Monthly"),
    ("Tourism", "Quarterly"),
]

METHOD_FOLDERS = {
    "Grasynda": "grasynda",
    "SeasonalMBB": "seasonalmbb",
    "Jittering": "jittering",
    "Scaling": "scaling",
    "TimeWarping": "timewarping",
    "TSMixup": "tsmixup",
    "MagnitudeWarping": "magnitudewarping",
    "DBA": "dba",
    "TimeVAE": "timevae",
    "TSDiff": "tsdiff",
}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    return value


def _relative_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("/", "\\")


def _target_sort_key(item: Dict[str, Any]) -> int:
    dataset_group = (item["Dataset"], item["Group"])
    try:
        return TARGETS.index(dataset_group)
    except ValueError:
        return len(TARGETS)


def _load_latest_target_rows(method_name: str, folder_name: str) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], list]:
    method_dir = RANDOM_SEARCH_ROOT / folder_name
    selected: Dict[Tuple[str, str], Dict[str, Any]] = {}
    warnings = []

    if not method_dir.exists():
        warnings.append(f"Missing method directory: {method_dir}")
        return selected, warnings

    for path in method_dir.glob("*_best_config.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"Could not read {path.name}: {exc}")
            continue

        dataset = row.get("Dataset")
        group = row.get("Group")
        params = row.get("Best_Params_For_Experiment") or row.get("Best_Params")
        if not dataset or not group or not isinstance(params, dict):
            warnings.append(f"Skipping malformed best-config file: {path.name}")
            continue

        target_key = (dataset, group)
        candidate = {
            "Method": method_name,
            "Dataset": dataset,
            "Group": group,
            "Params": _normalize_value(params),
            "Best_CV_MASE": row.get("Best_CV_MASE"),
            "Baseline_CV_MASE": row.get("Baseline_CV_MASE"),
            "Holdout_Best_MASE": row.get("Holdout_Best_MASE"),
            "Holdout_Baseline_MASE": row.get("Holdout_Baseline_MASE"),
            "Trials_Evaluated": row.get("Trials_Evaluated"),
            "Source_Best_Config_Path": _relative_path(path),
            "Source_Last_Modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            "_mtime": path.stat().st_mtime,
        }

        current = selected.get(target_key)
        if current is None or candidate["_mtime"] > current["_mtime"]:
            selected[target_key] = candidate

    return selected, warnings


def build_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "Forecast_Model": "NHITS",
        "Generated_At": datetime.now().isoformat(timespec="seconds"),
        "Target_Dataset_Groups": [
            {"Dataset": dataset, "Group": group}
            for dataset, group in TARGETS
        ],
        "Methods": {},
        "Completed_Methods": [],
        "Partial_Methods": [],
        "Missing_Methods": [],
        "Warnings": [],
    }

    for method_name, folder_name in METHOD_FOLDERS.items():
        selected_rows, warnings = _load_latest_target_rows(method_name, folder_name)
        payload["Warnings"].extend(warnings)

        target_rows = []
        missing_targets = []
        for dataset, group in TARGETS:
            row = selected_rows.get((dataset, group))
            if row is None:
                missing_targets.append({"Dataset": dataset, "Group": group})
                continue
            row = dict(row)
            row.pop("_mtime", None)
            target_rows.append(row)

        target_rows.sort(key=_target_sort_key)
        method_payload = {
            "Method": method_name,
            "Completed_Target_Count": len(target_rows),
            "Missing_Targets": missing_targets,
            "Targets": target_rows,
        }
        payload["Methods"][method_name] = method_payload

        if len(target_rows) == len(TARGETS):
            payload["Completed_Methods"].append(method_name)
        elif len(target_rows) > 0:
            payload["Partial_Methods"].append(method_name)
        else:
            payload["Missing_Methods"].append(method_name)

    return payload


def main() -> None:
    payload = build_payload()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved final NHITS optimal parameters: {OUTPUT_PATH}")
    print(f"Completed methods: {', '.join(payload['Completed_Methods']) or 'None'}")
    print(f"Partial methods: {', '.join(payload['Partial_Methods']) or 'None'}")
    print(f"Missing methods: {', '.join(payload['Missing_Methods']) or 'None'}")

    for method_name, method_payload in payload["Methods"].items():
        print(
            f"{method_name}: "
            f"{method_payload['Completed_Target_Count']}/{len(TARGETS)} targets"
        )


if __name__ == "__main__":
    main()
