"""Small report wrapper for Phase 3 generated metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("training/model3_phase3/results"))
    args = parser.parse_args()
    table = pd.read_csv(args.results / "final_comparison.csv")
    print(table.to_string(index=False))
    print("\nNo production model was exported by this evaluation.")


if __name__ == "__main__":
    main()