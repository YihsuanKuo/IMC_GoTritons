#!/usr/bin/env python3
"""
Test one fixed Aether Crystal portfolio over 100 official settlements.

This script reuses the existing GBM path generation and payoff logic from
aether_fair_value_simulator.py.
"""

from __future__ import annotations

import shutil
import os
from pathlib import Path

MPLCONFIGDIR = Path("/tmp/matplotlib-aether")
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aether_fair_value_simulator import (
    CONTRACT_SIZE,
    OUTPUT_DIR,
    Position,
    build_payoffs,
    fair_values_from_payoffs,
    portfolio_pnl,
    simulate_gbm_paths,
)


N_OFFICIAL_SETTLEMENTS = 100
PATHS_PER_SETTLEMENT = 100


CHOOSER_SECTION = [
    Position("AC_50_CO", "SELL", 50, 22.20),
    Position("AC_50_C", "BUY", 25, 12.05),
    Position("AC_50_P_2", "BUY", 25, 9.75),
    Position("AC_50_P", "BUY", 25, 12.05),
    Position("AC_50_C_2", "BUY", 25, 9.75),
]

BP_SECTION = [
    Position("AC_40_BP", "SELL", 25, 5.00),
    Position("AC_40_P", "BUY", 50, 6.55),
    Position("AC_35_P", "SELL", 50, 4.33),
]

KO_SECTION = [
    Position("AC_45_KO", "BUY", 150, 0.175),
    Position("AC_45_P", "SELL", 50, 9.05),
    Position("AC", "SELL", 50, 49.975),
]

USED_PRODUCTS = [
    "AC",
    "AC_35_P",
    "AC_40_BP",
    "AC_40_P",
    "AC_45_KO",
    "AC_45_P",
    "AC_50_C",
    "AC_50_CO",
    "AC_50_C_2",
    "AC_50_P",
    "AC_50_P_2",
]


def reset_output_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_settlements() -> pd.DataFrame:
    master_rng = np.random.default_rng()
    rows = []

    for settlement_id in range(1, N_OFFICIAL_SETTLEMENTS + 1):
        settlement_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        paths = simulate_gbm_paths(PATHS_PER_SETTLEMENT, seed=settlement_seed)
        fair_values = fair_values_from_payoffs(build_payoffs(paths))

        chooser_pnl = portfolio_pnl(CHOOSER_SECTION, fair_values)
        bp_pnl = portfolio_pnl(BP_SECTION, fair_values)
        ko_pnl = portfolio_pnl(KO_SECTION, fair_values)
        total_pnl = chooser_pnl + bp_pnl + ko_pnl

        row = {
            "settlement_id": settlement_id,
            "chooser_pnl": chooser_pnl,
            "bp_pnl": bp_pnl,
            "ko_pnl": ko_pnl,
            "total_pnl": total_pnl,
        }
        for product in USED_PRODUCTS:
            row[f"fair_value_{product}"] = fair_values[product]
        rows.append(row)

    columns = (
        ["settlement_id"]
        + [f"fair_value_{product}" for product in USED_PRODUCTS]
        + ["chooser_pnl", "bp_pnl", "ko_pnl", "total_pnl"]
    )
    return pd.DataFrame(rows, columns=columns)


def summarize_settlements(settlements: pd.DataFrame) -> pd.DataFrame:
    total_pnl = settlements["total_pnl"]
    profitable_count = int((total_pnl > 0).sum())
    return pd.DataFrame([{
        "mean_total_pnl": float(total_pnl.mean()),
        "std_total_pnl": float(total_pnl.std(ddof=1)),
        "min_total_pnl": float(total_pnl.min()),
        "max_total_pnl": float(total_pnl.max()),
        "p05_total_pnl": float(np.percentile(total_pnl, 5)),
        "median_total_pnl": float(np.percentile(total_pnl, 50)),
        "p95_total_pnl": float(np.percentile(total_pnl, 95)),
        "profitable_count": profitable_count,
        "profit_probability": profitable_count / N_OFFICIAL_SETTLEMENTS,
        "average_chooser_pnl": float(settlements["chooser_pnl"].mean()),
        "average_bp_pnl": float(settlements["bp_pnl"].mean()),
        "average_ko_pnl": float(settlements["ko_pnl"].mean()),
    }])


def plot_total_pnl_histogram(settlements: pd.DataFrame, out_dir: Path) -> None:
    total_pnl = settlements["total_pnl"]
    plt.figure(figsize=(8, 5))
    plt.hist(total_pnl, bins=20, edgecolor="black")
    plt.axvline(total_pnl.mean(), linestyle="--", linewidth=2, label=f"mean = {total_pnl.mean():,.0f}")
    plt.axvline(0, linestyle=":", linewidth=2, label="breakeven")
    plt.title("Fixed portfolio total PnL across 100 official settlements")
    plt.xlabel("Total PnL")
    plt.ylabel("Settlement count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fixed_portfolio_total_pnl_hist.png", dpi=160)
    plt.close()


def main() -> None:
    reset_output_folder(OUTPUT_DIR)

    settlements = run_settlements()
    summary = summarize_settlements(settlements)

    settlements.to_csv(OUTPUT_DIR / "fixed_portfolio_100_official_settlements.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "fixed_portfolio_100_official_summary.csv", index=False)
    plot_total_pnl_histogram(settlements, OUTPUT_DIR)

    stats = summary.iloc[0]
    made_money_more_than_50 = int(stats["profitable_count"]) > 50

    print("\nSaved outputs to:", OUTPUT_DIR.resolve())
    print(f"Profit probability: {stats['profit_probability']:.1%} ({int(stats['profitable_count'])}/{N_OFFICIAL_SETTLEMENTS})")
    print(f"Mean total PnL: {stats['mean_total_pnl']:,.2f}")
    print(f"Worst settlement PnL: {stats['min_total_pnl']:,.2f}")
    print(f"Best settlement PnL: {stats['max_total_pnl']:,.2f}")
    print(f"5th percentile PnL: {stats['p05_total_pnl']:,.2f}")
    print(f"Made money in more than 50 of 100 official settlements: {made_money_more_than_50}")


if __name__ == "__main__":
    main()
