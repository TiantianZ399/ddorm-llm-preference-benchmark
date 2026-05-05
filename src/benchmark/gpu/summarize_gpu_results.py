from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect GPU benchmark metrics into a CSV table.")
    parser.add_argument("--input_glob", default="outputs/gpu/**/eval*.json")
    parser.add_argument("--output_csv", default="outputs/gpu/results_summary.csv")
    return parser.parse_args()


def flatten(prefix: str, obj: dict[str, Any], out: dict[str, Any]) -> None:
    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flatten(name, value, out)
        else:
            out[name] = value


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path().glob(args.input_glob)):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        row: dict[str, Any] = {"path": str(path)}
        flatten("", payload, row)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
